#include "collectiveRlTransport.h"

#include <buzz/buzzvm.h>

/****************************************/
/****************************************/

static const std::string FB_CONTROLLER = "fgc";
static const Real WALL_THICKNESS            = 0.2;  // m
static const Real CYLINDER_RADIUS           = 0.5;  // m
static const Real CYLINDER_HEIGHT           = 0.25; // m
static const Real CYLINDER_MASS             = 100;  // kg
static const Real FOOTBOT_RADIUS            = 0.085036758f; // m
static const Real ROBOT_CYLINDER_DISTANCE   = 0.6; // m
static const Real CYLINDER_PLACEMENT_RADIUS = WALL_THICKNESS + ROBOT_CYLINDER_DISTANCE + FOOTBOT_RADIUS;

/****************************************/
/****************************************/

void CCollectiveRLTransport::Init(TConfigurationNode& t_tree) {
   /* Parse XML tree */
   GetNodeAttribute(t_tree, "data_file",       m_strOutFile);
   GetNodeAttribute(t_tree, "num_robots",      m_unNumRobots);
   GetNodeAttribute(t_tree, "goal",            m_cGoal);
   GetNodeAttribute(t_tree, "threshold",       m_fThreshold);
   GetNodeAttribute(t_tree, "obs_size",        m_unObsSize);
   GetNodeAttribute(t_tree, "action_size",     m_unActionSize);
   GetNodeAttribute(t_tree, "num_episodes",    m_unNumEpisodes);
   GetNodeAttribute(t_tree, "episode_time",    m_unEpisodeTime);
   GetNodeAttribute(t_tree, "time_out_reward", m_fTimeOutReward);
   GetNodeAttribute(t_tree, "goal_reward",     m_fGoalReward);
   m_bReachedGoal = false;
   m_unEpisodeCounter = 0;
   m_unEpisodeTicksLeft = m_unEpisodeTime;
   /* Connect to PyTorch */
   // TODO
   //m_pcPyTorch = new ModelServerClient(55555, 55555, m_unNumRobots, m_unObsSize, m_unActionSize);
   /* Create a new RNG */
   m_pcRNG = CRandom::CreateRNG("argos");
   /* Create and place stuff */
   CreateEntities();
   PlaceEntities(0);
   /* Call buzz Init() (HAS TO BE THE LAST LINE)*/
   CBuzzLoopFunctions::Init(t_tree);
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::CreateEntities() {
   /* Create the cylinder */
   m_pcCylinder = new CCylinderEntity(
      "c1",
      CVector3(),
      CQuaternion(),
      true,
      CYLINDER_RADIUS,
      CYLINDER_HEIGHT,
      CYLINDER_MASS);
   AddEntity(*m_pcCylinder);
   /* Create robots */
   CRadians cSlice = CRadians::TWO_PI / m_unNumRobots;
   std::ostringstream cFBId;
   CFootBotEntity* pcFB;
   CVector3 cPos;
   for(size_t i = 0; i < m_unNumRobots; ++i) {
      cFBId.str("");
      cFBId << "fb" << i;
      cPos.FromSphericalCoords(ROBOT_CYLINDER_DISTANCE,
                               CRadians::PI_OVER_TWO,
                               i * cSlice);
      cPos.SetZ(0.0);
      pcFB = new CFootBotEntity(
         cFBId.str(),
         FB_CONTROLLER,
         cPos,
         CQuaternion(-cSlice, CVector3::Z)
         );
      m_vecRobots.push_back(pcFB);
      AddEntity(*pcFB);
   }
   /* Generating random positions for the cylinder */
   /* We divide the arena in two horizontal halves */
   CRange<Real> cXRange(
      GetSpace().GetArenaLimits().GetMin().GetX() + CYLINDER_PLACEMENT_RADIUS,
      -CYLINDER_PLACEMENT_RADIUS
      );
   CRange<Real> cYRange(
      GetSpace().GetArenaLimits().GetMin().GetY() + CYLINDER_PLACEMENT_RADIUS,
      GetSpace().GetArenaLimits().GetMax().GetY() - CYLINDER_PLACEMENT_RADIUS
      );
   for(size_t i = 0; i < m_unNumEpisodes; ++i){
      cPos.Set(m_pcRNG->Uniform(cXRange),
               m_pcRNG->Uniform(cYRange),
               0.0);
      m_vecCylinderPos.push_back(cPos);
   }
   /* Generating random positions for the robots */
   for(size_t i = 0; i < m_unNumEpisodes; ++i) {
      CRadians cOffset = m_pcRNG->Uniform(CRadians::SIGNED_RANGE);
      m_vecRobotPos.push_back(std::vector<CVector3>());
      m_vecRobotOrient.push_back(std::vector<CQuaternion>());
      for(size_t j = 0; j < m_unNumRobots; ++j) {
         /* Create offset wrt cylinder center */
         CVector3 cPos(
            CVector3(ROBOT_CYLINDER_DISTANCE,
                     CRadians::PI_OVER_TWO,
                     j * cSlice + cOffset));
         /* Translate to actual cylinder center */
         cPos += m_vecCylinderPos[i];
         /* Add position */
         m_vecRobotPos.back().push_back(cPos);
         /* Calculate orientation to cylinder center */
         CQuaternion cOrient(j * cSlice + cOffset + CRadians::PI, CVector3::Z);
         m_vecRobotOrient.back().push_back(cOrient);
      }
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PlaceEntities(UInt32 un_episode) {
   /* Make sure the episode is valid */
   if(un_episode >= m_unNumEpisodes) {
      THROW_ARGOSEXCEPTION("Episode " << un_episode << " is beyond the maximum of " << m_unNumEpisodes);
   }
   /* The placements we chose are collision-free by construction, no need to
    * check for collisions */
   MoveEntity(m_pcCylinder->GetEmbodiedEntity(), // body
              m_vecCylinderPos[un_episode],      // position
              CQuaternion(),                     // orientation
              false,                             // not a check
              true);                             // ignore collisions
   for(size_t i = 0; i < m_vecRobots.size(); ++i) {
      MoveEntity(m_vecRobots[i]->GetEmbodiedEntity(), // body
                 m_vecRobotPos[un_episode][i],        // position
                 m_vecRobotOrient[un_episode][i],     // orientation
                 false,                               // not a check
                 true);                               // ignore collisions
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Reset() {
   PlaceEntities(m_unEpisodeCounter);
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Destroy() {
   // TODO
   //delete m_pcPyTorch;
}

/****************************************/
/****************************************/

struct PutIncreases : public CBuzzLoopFunctions::COperation {
   
   PutIncreases(std::vector<Real>& vec_l_increase,
                std::vector<Real>& vec_r_increase) :
      LIncrease(vec_l_increase),
      RIncrease(vec_r_increase) {}
   
   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {
      BuzzPut(t_vm, "L_increase", static_cast<float>(LIncrease[t_vm->robot]));
      BuzzPut(t_vm, "R_increase", static_cast<float>(RIncrease[t_vm->robot]));
      DEBUG("[Ex] [t=%u] R#%u A = %f,%f\n",
            CSimulator::GetInstance().GetSpace().GetSimulationClock(),
            t_vm->robot,
            LIncrease[t_vm->robot],
            RIncrease[t_vm->robot]);
   }
   
   std::vector<Real> LIncrease;
   std::vector<Real> RIncrease;
};

/****************************************/
/****************************************/

struct GetWheelSpeeds : public CBuzzLoopFunctions::COperation {
   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {
      LWheels.push_back(buzzobj_getfloat(BuzzGet(t_vm, "L_wheel")));
      RWheels.push_back(buzzobj_getfloat(BuzzGet(t_vm, "R_wheel")));
   }
   std::vector<float> LWheels;
   std::vector<float> RWheels;
};


void CCollectiveRLTransport::GetObservations(std::vector<float>& vec_obs,
                                             EEpisodeState e_state){
   for(size_t i = 0; i < m_vecRobots.size(); ++i) {
      /* Get robot pose */
      CVector3& cRobotPos =
         m_vecRobots[i]->GetEmbodiedEntity().GetOriginAnchor().Position;
      CQuaternion& cRobotOrient =
         m_vecRobots[i]->GetEmbodiedEntity().GetOriginAnchor().Orientation;
      CRadians cRobotZ, cRobotY, cRobotX;
      cRobotOrient.ToEulerAngles(cRobotZ, cRobotY, cRobotX);
      /* Get vector from robot to goal (robot-local) */
      CVector2 cVecRobot2Goal(
         m_cGoal.GetX() - cRobotPos.GetX(),
         m_cGoal.GetY() - cRobotPos.GetY());
      cVecRobot2Goal.Rotate(-cRobotZ);
      /* Get vector from robot to cylinder (robot-local) */
      CVector3& cCylinderPos =
         m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position;
      CVector2 cVecRobot2Cylinder(
         cCylinderPos.GetX() - cRobotPos.GetX(),
         cCylinderPos.GetY() - cRobotPos.GetY());
      cVecRobot2Cylinder.Rotate(-cRobotZ);
      /* Get vector from cylinder to goal (robot-local) */
      CVector2 cVecCylinder2Goal =
         cVecRobot2Goal - cVecRobot2Cylinder;
      /* Calculate reward and done state */
      Real fReward;
      Real fDone;
      switch(e_state) {
         case EPISODE_RUNNING: {
            fReward = -1.0 + (1.0 / (10.0 * cVecRobot2Goal.Length()));
            fDone = 0.0;
            break;
         }
         case EPISODE_SUCCESS: {
            fReward = m_fGoalReward;
            fDone = 1.0;
            break;
         }
         case EPISODE_TIMEOUT: {
            fReward = m_fTimeOutReward;
            fDone = 1.0;
            break;
         }
      }
      /* Get the wheel speeds*/
      GetWheelSpeeds cGWS;
      BuzzForeachVM(cGWS);
      Real fLWheel = cGWS.LWheels[i];
      Real fRWheel = cGWS.RWheels[i];
      /* Store the observations */
      vec_obs[i * m_unObsSize + 0] = cVecRobot2Goal.Length();
      vec_obs[i * m_unObsSize + 1] = ToDegrees(cVecRobot2Goal.Angle()).GetValue();
      vec_obs[i * m_unObsSize + 2] = fLWheel;
      vec_obs[i * m_unObsSize + 3] = fRWheel;
      vec_obs[i * m_unObsSize + 4] = cVecRobot2Cylinder.Length();
      vec_obs[i * m_unObsSize + 5] = ToDegrees(cVecRobot2Cylinder.Angle()).GetValue();
      vec_obs[i * m_unObsSize + 6] = cVecCylinder2Goal.Length();
      vec_obs[i * m_unObsSize + 7] = fDone;
      vec_obs[i * m_unObsSize + 8] = fReward;
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PreStep() {   
   /************************************
    This function will grab the wheel speeds for each robot
    for the current time step.
   ************************************/
   /*
     Observation:
     for each robot:
     Vector between robot and goal
     Left and Right wheel speeds
     Distance from the cylinder to the goal
     Vector from robot to cylinder
   */
   std::vector<float> vecObs(m_unNumRobots * m_unObsSize);
   GetObservations(vecObs, EPISODE_RUNNING);
   std::string OBS_DESCRIPTIONS[] = {
      "robot2goal_dist",
      "robot2goal_angle",
      "lwheel",
      "rwheel",
      "robot2cylinder_dist",
      "robot2cylinder_angle",
      "cylinder2goal_dist",
      "done",
      "reward"
   };
   for(size_t i = 0; i < m_unNumRobots; ++i) {
      for(size_t j = 0; j < m_unObsSize; ++j) {
         DEBUG("[E%u] [t=%u] [R=%zu] %s = %f\n",
               m_unEpisodeCounter,
               GetSpace().GetSimulationClock(),
               i,
               OBS_DESCRIPTIONS[j].c_str(),
               vecObs[i * m_unObsSize + j]);
      }
   }
   DEBUG("\n");
   /* Send observations to PyTorch */
   // TODO m_pcPyTorch->SendAgentObservations(vecObs);
   /* Get actions from PyTorch */
   // TODO float* pfActions = m_pcPyTorch->GetAgentActions();
   /* === PLACEHOLDER CODE STARTS === */
   float* pfActions = new float[m_unNumRobots * m_unActionSize];
   for(size_t i = 0; i < m_unNumRobots; ++i) {
      float* pfAction = pfActions + i * m_unActionSize;
      pfAction[0] = GetSpace().GetSimulationClock() % 3;
      pfAction[1] = (GetSpace().GetSimulationClock() + 1) % 3;
   }
   for(size_t i = 0; i < m_unNumRobots; ++i) {
      float* pfAction = pfActions + i * m_unActionSize;
      DEBUG("[E%u] [t=%u] [R=%zu] RAW A = %f,%f\n",
            m_unEpisodeCounter,
            GetSpace().GetSimulationClock(),
            i,
            pfAction[0],
            pfAction[1]);
   }
   /* === PLACEHOLDER CODE ENDS === */
   std::vector<Real> vecLIncrease(m_unNumRobots);
   std::vector<Real> vecRIncrease(m_unNumRobots);
   for(size_t i = 0; i < m_vecRobots.size(); ++i) {
      float* pfAction = pfActions + i * m_unActionSize;
      if(pfAction[0] == 0.0)      vecLIncrease[i] = -0.1;
      else if(pfAction[0] == 1.0) vecLIncrease[i] =  0.0;
      else if(pfAction[0] == 2.0) vecLIncrease[i] =  0.1;
      if(pfAction[1] == 0.0)      vecRIncrease[i] = -0.1;
      else if(pfAction[1] == 1.0) vecRIncrease[i] =  0.0;
      else if(pfAction[1] == 2.0) vecRIncrease[i] =  0.1;
   }
   BuzzForeachVM(PutIncreases(vecLIncrease, vecRIncrease));
   /* Cleanup */
   delete[] pfActions;
}

/****************************************/
/****************************************/

bool CCollectiveRLTransport::IsExperimentFinished() {
   /* This is where we will check if the object breaks
      or if we have reached the goal position. */
   if(m_unEpisodeCounter < m_unNumEpisodes) {
      return false;
   }
   LOG << "Ending Experiment" << std::endl;
   return true;
}

/****************************************/
/****************************************/

/**
 * Executes user-defined logic when the experiment finishes.
 * This method is called within CSimulator::IsExperimentFinished()
 * as soon as its return value evaluates to <tt>true</tt>. This
 * method is executed before Destroy().
 * You can use this method to perform final calculations at the
 * end of an experiment.
 * The default implementation of this method does nothing.
 */
void CCollectiveRLTransport::PostExperiment() {
   // LOG<<"Closing the Server"<<std::endl;
   // m_pcPyTorch->CloseServer();
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PostStep() {
   /* Decrement remaining time */
   --m_unEpisodeTicksLeft;
   /* Check if the cylinder reached the goal */
   CVector3& cCylinderPos = m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position;
   CVector2 cCylinder2Goal(
      m_cGoal.GetX() - cCylinderPos.GetX(),
      m_cGoal.GetY() - cCylinderPos.GetY());
   m_bReachedGoal = (cCylinder2Goal.Length() < m_fThreshold);
   /* If we haven't reached our experiment limit then reset */
   if(IsEpisodeFinished()) {
      LOG << "Episode " << m_unEpisodeCounter << " is done" << std::endl;
      std::vector<float> vecObs(m_unNumRobots * m_unObsSize);
      EEpisodeState eState = m_bReachedGoal ? EPISODE_SUCCESS : EPISODE_TIMEOUT;
      GetObservations(vecObs, eState);
      std::string OBS_DESCRIPTIONS[] = {
         "robot2goal_dist",
         "robot2goal_angle",
         "lwheel",
         "rwheel",
         "robot2cylinder_dist",
         "robot2cylinder_angle",
         "cylinder2goal_dist",
         "done",
         "reward"
      };
      for(size_t i = 0; i < m_unNumRobots; ++i) {
         for(size_t j = 0; j < m_unObsSize; ++j) {
            DEBUG("[E%u] [t=%u] %s = %f\n", m_unEpisodeCounter, GetSpace().GetSimulationClock(), OBS_DESCRIPTIONS[j].c_str(), vecObs[i * m_unObsSize + j]);
         }
      }
      DEBUG("\n");
      /* Send observations to PyTorch */
      // TODO m_pcPyTorch->SendAgentObservations(observations);
      /* Restart episode */
      ++m_unEpisodeCounter;
      if(m_unEpisodeCounter < m_unNumEpisodes) {
         m_unEpisodeTicksLeft = m_unEpisodeTime;
         GetSimulator().Reset();
      }
   }
}

/****************************************/
/****************************************/

bool CCollectiveRLTransport::IsEpisodeFinished() {
   if(m_bReachedGoal) {
      LOG << "Reached Goal" << std::endl;
      m_bReachedGoal = false;
      return true;
   }
   if(m_unEpisodeTicksLeft == 0) {
      LOG << "We have timed out" << std::endl;
      return true;
   }
   return false;
}

/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(CCollectiveRLTransport, "collective_rl_transport");
