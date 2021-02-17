#include "collectiveRlTransport.h"
#include <buzz/buzzvm.h>
#include <cmath>

using namespace argos;

/****************************************/
/****************************************/

static const std::string FB_CONTROLLER = "fgc";
static const Real WALL_THICKNESS            = 0.2;  // m
static const Real CYLINDER_RADIUS           = 0.5;  // m
static const Real CYLINDER_HEIGHT           = 0.25; // m
static const Real CYLINDER_MASS             = 100;  // kg
static const Real OBSTACLE_RADIUS           = 0.5;  // m
static const Real OBSTACLE_HEIGHT           = 0.5;  // m
static const Real OBSTACLE_MASS             = 100;  // kg
static const Real FOOTBOT_RADIUS            = 0.085036758f; // m
static const Real ROBOT_CYLINDER_DISTANCE   = 0.6;  // m
static const Real CYLINDER_PLACEMENT_RADIUS = WALL_THICKNESS + ROBOT_CYLINDER_DISTANCE + FOOTBOT_RADIUS;

static const std::string OBS_DESCRIPTIONS[] = {
   "robot2goal_dist",
   "robot2goal_angle",
   "lwheel",
   "rwheel",
   "robot2cylinder_dist",
   "robot2cylinder_angle",
   "cylinder2goal_dist",
   "reward"
};

static const std::string ACTIONS_DESCRIPTIONS[] = {
   "lwheel",
   "rwheel"
};

/****************************************/
/****************************************/

CCollectiveRLTransport::CCollectiveRLTransport() :
   m_unNumObs(7+24),
   m_unNumActions(3){
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Init(TConfigurationNode& t_tree) {
   try {
      /* Parse XML tree */
      LOG<<"Initiating"<<std::endl;
      GetNodeAttribute(t_tree, "data_file",       m_strOutFile);
      GetNodeAttribute(t_tree, "num_robots",      m_unNumRobots);
      GetNodeAttribute(t_tree, "max_robot_failures", m_unMaxRobotFailures);
      GetNodeAttribute(t_tree, "latest_failure_time", m_unLatestFailureTime);
      GetNodeAttribute(t_tree, "chance_failure", m_fChanceFailure);
      GetNodeAttribute(t_tree, "goal",            m_cGoal);
      GetNodeAttribute(t_tree, "threshold",       m_fThreshold);
      GetNodeAttribute(t_tree, "num_episodes",    m_unNumEpisodes);
      GetNodeAttribute(t_tree, "episode_time",    m_unEpisodeTime);
      GetNodeAttribute(t_tree, "time_out_reward", m_fTimeOutReward);
      GetNodeAttribute(t_tree, "threshold_freq", m_unDecThresholdTime);
      GetNodeAttribute(t_tree, "threshold_dec", m_fDecThreshold);
      GetNodeAttribute(t_tree, "min_threshold", m_fMinThreshold);
      GetNodeAttribute(t_tree, "goal_reward",     m_fGoalReward);
      GetNodeAttribute(t_tree, "socket_port",     m_unPort);
      GetNodeAttribute(t_tree, "alphabet_size", m_unAlphabetSize);
      GetNodeAttribute(t_tree, "proximity_range", m_fProximityRange);
      GetNodeAttribute(t_tree, "num_obstacles", m_unNumObstacles);
      GetNodeAttribute(t_tree, "use_base_model", m_unBaseModel);

      /* Footbot dynamic equation parameters*/
      m_fFootbotAxelLength = 0.14; // m
      m_fFootbotWheelRadius = 0.029112741; // m

      /* Stats to be sent to Data: Force vector (direction and magnitude) for every robot*/
      m_unNumStats = 2;
      /*
       * Connect to PyTorch
       */
      /* Connect to Server */
      LOG<<m_unPort<<std::endl;
      socket.Connect("localhost", m_unPort);

      /* Send parameters */
      SocketSendParams();
      LOG << "[INFO] Connection to PyTorch server successful" << std::endl;
      /* Initialize episode-related variables */
      m_bReachedGoal = false;
      m_unEpisodeCounter = 0;
      m_unEpisodeTicksLeft = m_unEpisodeTime;
      /* Create structures for observations, reward, and actions */
      m_vecObs.resize(m_unNumObs * m_unNumRobots, 0.0);
      m_vecFailures.resize(m_unNumRobots, 0);
      m_vecRewards.resize(m_unNumRobots, 0.0);
      m_vecStats.resize(m_unNumRobots * m_unNumStats, 0.0);
      m_vecActions.resize(m_unNumActions * m_unNumRobots, 0.0);
      /* Create a new RNG */
      m_pcRNG = CRandom::CreateRNG("argos");
      /* Create and place stuff */
      CreateEntities();
      LOG<<"Added Entities\n";
      //Print first failure set
      for(size_t i = 0; i < m_unNumRobots; i++){
        LOG << m_vecRobotFailures[m_unEpisodeCounter][i]<<" ";
      }
      LOG<<std::endl;
      PlaceEntities(0);
      LOG<<"Placed Entities\n";
      /* Call buzz Init() (HAS TO BE THE LAST LINE) */
      CBuzzLoopFunctions::Init(t_tree);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("while initializing the loop functions", ex);
   }
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
      /** Need to chage the range of the proximity sensor
          If m_fProximityRange = 0 then the sensor range will
          stay at default */
      AddEntity(*pcFB);
      if(m_fProximityRange > 0.0){
        CProximitySensorEquippedEntity& cPSEE = pcFB->GetProximitySensorEquippedEntity();
        CProximitySensorEquippedEntity::SSensor::TList& listPS = cPSEE.GetSensors();
        for(auto itSensor = listPS.begin(); itSensor != listPS.end(); ++itSensor){
          (*itSensor)->Direction.Normalize();
          (*itSensor)->Direction *= m_fProximityRange;
        }
      }
   }
   /** Create Cylinder Obstacles */
   std::ostringstream cCID;
   CCylinderEntity* pcC;
   for(size_t i = 0; i < m_unNumObstacles; ++i){
     cCID.str("");
     cCID << "o" << i;
     pcC = new CCylinderEntity(
        cCID.str(),
        CVector3(),
        CQuaternion(),
        false,
        OBSTACLE_RADIUS,
        OBSTACLE_HEIGHT,
        OBSTACLE_MASS);
     m_vecObstacles.push_back(pcC);
     AddEntity(*pcC);
   }
   /* Generating random positions for the cylinder */
   /* We divide the arena in two horizontal halves */
   CRange<Real> cXCylinderRange(
      GetSpace().GetArenaLimits().GetMin().GetX() + CYLINDER_PLACEMENT_RADIUS,
      GetSpace().GetArenaLimits().GetMin().GetX()/2 -CYLINDER_PLACEMENT_RADIUS
      );
   CRange<Real> cXObstacleRange(
      GetSpace().GetArenaLimits().GetMin().GetX()/2,
      0
   );
   CRange<Real> cYRange(
      GetSpace().GetArenaLimits().GetMin().GetY() + CYLINDER_PLACEMENT_RADIUS,
      GetSpace().GetArenaLimits().GetMax().GetY() - CYLINDER_PLACEMENT_RADIUS
      );
   for(size_t i = 0; i < m_unNumEpisodes; ++i){
      cPos.Set(m_pcRNG->Uniform(cXCylinderRange),
               m_pcRNG->Uniform(cYRange),
               0.0);
      m_vecCylinderPos.push_back(cPos);
      //Generate Failure Times for all episodes
      m_vecRobotFailures.push_back(GenerateRobotFailure());
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
   /** Generate Random Positions for the obstacles */
   for(size_t i = 0; i < m_unNumEpisodes; i++){
     m_vecObstaclePos.push_back(std::vector<CVector3>());
     for(size_t j = 0; j < m_unNumObstacles; j++){
       cPos.Set(m_pcRNG->Uniform(cXObstacleRange),
                m_pcRNG->Uniform(cYRange),
                0.0);
       m_vecObstaclePos.back().push_back(cPos);
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
   /* Get the old position of the cylinder*/
   m_cOldCylinderPos = m_vecCylinderPos[un_episode];
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
   for(size_t i = 0; i < m_vecObstacles.size(); ++i){
     MoveEntity(m_vecObstacles[i]->GetEmbodiedEntity(), // body
                m_vecObstaclePos[un_episode][i],        // position
                CQuaternion(),                          // orientation
                false,                                  // not a check
                true);                                  // ignore collisions
   }
}

/****************************************/
/****************************************/

std::vector<SInt32> CCollectiveRLTransport::GenerateRobotFailure(){
  /*
    Mutates m_vecRobotFailures to have a new set of values. The index in the vector corresponds to the
    number of a robot. The entry determines at which timestep the robot will fail. A time of -1 indicates
    that robot will not fail this experiment.
  */

  // Below is a sampling without replacement algorithm. Indices is a collection of all unused robot indices.
  std::vector<SInt32> failures;
  std::vector<UInt32> indices;
  UInt32 remainingIndices = m_unNumRobots;
  SInt32 *robotFailureTimes = (SInt32 *) malloc(sizeof(SInt32) * m_unNumRobots);
  for(size_t i = 0; i < m_unNumRobots; ++i) {
    indices.push_back(i);
    robotFailureTimes[i] = -1;
  }

  CRange<Real> probabilityRange = CRange<Real>(0.0, 1.0);
  CRange<UInt32> failureTimeRange = CRange<UInt32>(0, m_unLatestFailureTime);
  for(size_t i = 0; i < m_unMaxRobotFailures; ++i) {
    // Select a robot
    CRange<UInt32> robotsRange = CRange<UInt32>(0, remainingIndices);
    size_t chosenIndex = m_pcRNG->Uniform(robotsRange);
    size_t chosenRobot = indices[chosenIndex];
    // Remove the selected robot's index from the list of available choices
    auto iterator = indices.begin();
    std::advance(iterator, chosenIndex);
    indices.erase(iterator);
    remainingIndices--;
    // with m_fChanceFailure probability, assign the robot a failure time. Otherwise it will not fail.
    if (m_pcRNG->Uniform(probabilityRange) < m_fChanceFailure) {
      robotFailureTimes[chosenRobot] = m_pcRNG->Uniform(failureTimeRange);
    } else {
      robotFailureTimes[chosenRobot] = -1;
    }
  }
  failures.assign(robotFailureTimes, robotFailureTimes + m_unNumRobots);
  return failures;

}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Reset() {
   for(size_t i = 0; i < m_unNumRobots; i++){
     LOG << m_vecRobotFailures[m_unEpisodeCounter][i]<<" ";
   }
   LOG<<std::endl;
   PlaceEntities(m_unEpisodeCounter);
   m_bReachedGoal = false;

}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Destroy() {

}

/****************************************/
/****************************************/

struct PutIncreases : public CBuzzLoopFunctions::COperation {

   PutIncreases(std::vector<Real>& vec_l_increase,
                std::vector<Real>& vec_r_increase,
                std::vector<UInt32>& vec_faiulure,
                std::vector<Real>& vec_angle_to_goal,
                std::vector<UInt32>& vec_base_model) :
      LIncrease(vec_l_increase),
      RIncrease(vec_r_increase),
      Failure(vec_faiulure),
      AngleToGoal(vec_angle_to_goal),
      BaseModel(vec_base_model) {}

   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {
      BuzzPut(t_vm, "L_increase", static_cast<float>(LIncrease[t_vm->robot]));
      BuzzPut(t_vm, "R_increase", static_cast<float>(RIncrease[t_vm->robot]));
      BuzzPut(t_vm, "failure", static_cast<int>(Failure[t_vm->robot]));
      BuzzPut(t_vm, "AngleToGoal", static_cast<float>(AngleToGoal[t_vm->robot]));
      BuzzPut(t_vm, "BaseModel", static_cast<int>(BaseModel[t_vm->robot]));
      /*DEBUG("[Ex] [t=%u] [R=%u] A = %f,%f F = %u\n",
            CSimulator::GetInstance().GetSpace().GetSimulationClock(),
            t_vm->robot,
            LIncrease[t_vm->robot],
            RIncrease[t_vm->robot],
            Failure[t_vm->robot]);*/
   }

   std::vector<Real> LIncrease;
   std::vector<Real> RIncrease;
   std::vector<UInt32> Failure;
   std::vector<Real> AngleToGoal;
   std::vector<UInt32> BaseModel;
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


void CCollectiveRLTransport::GetObservations(EEpisodeState e_state){
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
      CVector2 cVecCylinder2Goal
         (m_cGoal.GetX() - cCylinderPos.GetX(), m_cGoal.GetY() - cCylinderPos.GetY());
      /* Get the cosine similarity of the cylinder */
      CVector2 cMotion(cCylinderPos.GetX() - m_cOldCylinderPos.GetX(), cCylinderPos.GetY() - m_cOldCylinderPos.GetY());
      Real fDirection = 0;
      if(cMotion.Length()>1e-4){
        fDirection= cVecCylinder2Goal.DotProduct(cMotion) /
                          (cVecCylinder2Goal.Length()*cMotion.Length());

      }

      /* Check if the robot has failed */
      float hasFailed = 0;
      UInt32 ticksElapsed = m_unEpisodeTime - m_unEpisodeTicksLeft;
      if (m_vecRobotFailures[m_unEpisodeCounter][i] != -1 && m_vecRobotFailures[m_unEpisodeCounter][i] <= ticksElapsed) {
	       hasFailed = 1;
      }


      /* Calculate reward */
      Real fReward;
      switch(e_state) {
         case EPISODE_RUNNING: {
            //fReward = -1.0 + (1.0 / (10.0 * cVecRobot2Goal.Length()));
            /* Cost of living + direction x reward for moving */
            fReward = -2 + fDirection;
            break;
         }
         case EPISODE_SUCCESS: {
            fReward = m_fGoalReward;
            break;
         }
         case EPISODE_TIMEOUT: {
            fReward = m_fTimeOutReward;
            break;
         }
      }


      //DEBUG("cMotion = (%f,%f)\n", cMotion.GetX(), cMotion.GetY());
      //DEBUG("cVecCylinder2Goal = (%f,%f)\n", cVecCylinder2Goal.GetX(), cVecCylinder2Goal.GetY());
      //DEBUG("fDirection = %f\n", fDirection);
      /* Get the wheel speeds*/
      GetWheelSpeeds cGWS;
      BuzzForeachVM(cGWS);
      Real fLWheel = cGWS.LWheels[i];
      Real fRWheel = cGWS.RWheels[i];
      /* Store the observations */
      m_vecObs[i * m_unNumObs + 0] = cVecRobot2Goal.Length();
      m_vecObs[i * m_unNumObs + 1] = ToDegrees(cVecRobot2Goal.Angle()).GetValue();
      m_vecObs[i * m_unNumObs + 2] = fLWheel;
      m_vecObs[i * m_unNumObs + 3] = fRWheel;
      m_vecObs[i * m_unNumObs + 4] = cVecRobot2Cylinder.Length();
      m_vecObs[i * m_unNumObs + 5] = ToDegrees(cVecRobot2Cylinder.Angle()).GetValue();
      m_vecObs[i * m_unNumObs + 6] = cVecCylinder2Goal.Length();
      // Get the proximity sensor values
      const std::vector<argos::CCI_FootBotProximitySensor::SReading>& tReadings =
        m_vecRobots[i]->GetControllableEntity().GetController().GetSensor <CCI_FootBotProximitySensor> ("footbot_proximity")->GetReadings();
      for(size_t t = 0; t < tReadings.size(); t++){
        m_vecObs[i * m_unNumObs + 7 + t] = tReadings[t].Value;
      }
      // Failure flag must always be at the end of the observations
      m_vecFailures[i] = hasFailed;

      m_vecRewards[i] = fReward;
   }

}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PreStep() {
   GetObservations(EPISODE_RUNNING);
   CalculateRobotStats();
   m_cOldCylinderPos = m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position;
   /*for(size_t i = 0; i < m_unNumRobots; ++i) {
      for(size_t j = 0; j < m_unNumObs; ++j) {
         DEBUG("[E%u] [t=%u] [R=%zu] %s = %f\n",
               m_unEpisodeCounter,
               GetSpace().GetSimulationClock(),
               i,
               OBS_DESCRIPTIONS[j].c_str(),
               m_vecObs[i * m_unNumObs + j]);
      }
   }
   DEBUG("\n");*/
   /* Send observations to PyTorch */
   SocketSendEpisodeState(EPISODE_RUNNING);
   SocketSendObservations();
   SocketSendFailures();
   SocketSendRewards();
   SocketSendRobotStats();
   /* Get actions from PyTorch */
   SocketGetActions(b_Actions);

   /*for(size_t i = 0; i < m_unNumRobots; ++i) {
      float* pfAction = &m_vecActions[0] + i * m_unNumActions;
      DEBUG("[E%u] [t=%u] [R=%zu] RAW A = %f,%f\n",
            m_unEpisodeCounter,
            GetSpace().GetSimulationClock(),
            i,
            m_vecActions[i*3],
            m_vecActions[i*3+1]);
   }*/
   std::vector<Real> vecLIncrease(m_unNumRobots);
   std::vector<Real> vecRIncrease(m_unNumRobots);
   std::vector<UInt32> vecGripper(m_unNumRobots);
   std::vector<UInt32> vecFailure(m_unNumRobots);
   std::vector<Real> vecAngleToGoal(m_unNumRobots);
   std::vector<UInt32> vecBaseModel(m_unNumRobots);
   for(size_t i = 0; i < m_vecRobots.size(); ++i) {
      float* pfAction = &m_vecActions[0] + i * m_unNumActions;
      float* pfObs = &m_vecObs[0] + i * m_unNumObs;
      vecLIncrease[i] = pfAction[0];
      vecRIncrease[i] = pfAction[1];
      vecGripper[i] = pfAction[2];
      vecFailure[i] = m_vecFailures[i];
      vecAngleToGoal[i] = pfObs[1];
      vecBaseModel[i] = m_unBaseModel;
   }
   BuzzForeachVM(PutIncreases(vecLIncrease, vecRIncrease, vecGripper, vecAngleToGoal, vecBaseModel));
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
   LOG<<"Closing the Server"<<std::endl;
   SocketSendTermination();
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PostStep() {
   /* Decrement remaining time */
   --m_unEpisodeTicksLeft;
   /* Check if the cylinder reached the goal */
   m_bReachedGoal = CylinderAtTarget();
   EEpisodeState eState = EPISODE_RUNNING;
   /* If we haven't reached our experiment limit then reset */
   if(IsEpisodeFinished()) {
      LOG << "Episode " << m_unEpisodeCounter << " is done" << std::endl;
      /* check to see if we need to decrease the threshold */
      if((m_unEpisodeCounter+1) % m_unDecThresholdTime == 0){
        if(m_fThreshold > m_fMinThreshold){
          m_fThreshold = m_fThreshold - m_fDecThreshold;
          LOG << "Updating Threshold to:"<< m_fThreshold <<std::endl;
        }
      }

      eState = m_bReachedGoal ? EPISODE_SUCCESS : EPISODE_TIMEOUT;
      GetObservations(eState);
      CalculateRobotStats();
      /*for(size_t i = 0; i < m_unNumRobots; ++i) {
         for(size_t j = 0; j < m_unNumObs; ++j) {
            DEBUG("[E%u] [t=%u] %s = %f\n", m_unEpisodeCounter, GetSpace().GetSimulationClock(), OBS_DESCRIPTIONS[j].c_str(), m_vecObs[i * m_unNumObs + j]);
         }
      }
      DEBUG("\n");*/
      /* Send observations to PyTorch */
      SocketSendEpisodeState(eState);
      SocketSendObservations();
      SocketSendFailures();
      SocketSendRewards();
      SocketSendRobotStats();
      SocketGetAck();
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

bool CCollectiveRLTransport::CylinderAtTarget() {
   CVector3& cCylinderPos =
      m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position;
   CVector2 cCylinder2Goal(
      m_cGoal.GetX() - cCylinderPos.GetX(),
      m_cGoal.GetY() - cCylinderPos.GetY());
   return cCylinder2Goal.Length() < m_fThreshold;
}

/****************************************/
/****************************************/

bool CCollectiveRLTransport::IsEpisodeFinished() {
   if(m_bReachedGoal) {
      LOG << "Reached Goal" << std::endl;
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

void CCollectiveRLTransport::CalculateRobotStats(){
  for(size_t i = 0; i < m_vecRobots.size(); ++i) {
     /* Get robot pose */
     CVector3& cRobotPos =
        m_vecRobots[i]->GetEmbodiedEntity().GetOriginAnchor().Position;
     /* Get robot orientation*/
     CQuaternion& cRobotOrient =
        m_vecRobots[i]->GetEmbodiedEntity().GetOriginAnchor().Orientation;
     CRadians cRobotZ, cRobotY, cRobotX;
     cRobotOrient.ToEulerAngles(cRobotZ, cRobotY, cRobotX);
     /* Get the wheel speeds*/
     GetWheelSpeeds cGWS;
     BuzzForeachVM(cGWS);
     Real fLWheel = cGWS.LWheels[i];
     Real fRWheel = cGWS.RWheels[i];

     /*Calculate force vector*/
     Real deltaX = (m_fFootbotWheelRadius/2.0) * (fLWheel + fRWheel) * Cos(cRobotZ);
     Real deltaY = (m_fFootbotWheelRadius/2.0) * (fLWheel + fRWheel) * Sin(cRobotZ);

     Real magnitude = Sqrt((cRobotPos.GetX() - deltaX)*(cRobotPos.GetX() - deltaX)
                            + (cRobotPos.GetY() - deltaY)*(cRobotPos.GetY() - deltaY));

     m_vecStats[i * m_unNumStats + 0] = magnitude;
     m_vecStats[i * m_unNumStats + 1] = ToDegrees(cRobotZ).GetValue();
   }
}



/****************************************/
/****************************************/

void CCollectiveRLTransport::SocketSendEpisodeState(EEpisodeState e_state) {
   CByteArray b_Done;
   b_Done<<uint16_t(0);
   b_Done<<uint16_t(e_state !=EPISODE_RUNNING);
   b_Done<<uint16_t(m_bReachedGoal);
   socket.SendMsg(b_Done, true);
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::SocketSendTermination() {
   CByteArray b_Done;
   // (experiment done, episode done, no reward)
   b_Done << 1 << 0 << 0;
   socket.SendMsg(b_Done, false);
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::SocketSendParams() {
   /* Make the parameter buffer */
   CByteArray vecParams;
   vecParams << m_unNumRobots;
   vecParams << m_unNumObs;
   vecParams << m_unNumActions;
   vecParams << m_unNumStats;
   vecParams << m_unAlphabetSize;
   /*DEBUG("m_unNumRobots  = %u\n", m_unNumRobots);
   DEBUG("m_unNumObs     = %u\n", m_unNumObs);
   DEBUG("m_unNumActions = %u\n", m_unNumActions);*/
   /* Send the parameters */
   socket.SendMsg(vecParams, false);
   /* Wait for acknowledgment */
   SocketGetAck();
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::SocketSendObservations() {
   CByteArray b_vecObs;
   for(size_t i = 0; i < m_vecObs.size(); ++i){
     appendFloatIEEE754(b_vecObs, m_vecObs[i]);
   }
   socket.SendMsg(b_vecObs, true);
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::SocketSendFailures() {
   CByteArray b_vecFailures;
   for(size_t i = 0; i < m_vecFailures.size(); ++i){
     b_vecFailures << m_vecFailures[i];
   }
   socket.SendMsg(b_vecFailures, true);
}

/****************************************/
/****************************************/
void CCollectiveRLTransport::SocketSendRewards(){
  CByteArray b_vecRewards;
  for(size_t i = 0; i < m_vecRewards.size(); ++i){
    appendFloatIEEE754(b_vecRewards, m_vecRewards[i]);
  }
  socket.SendMsg(b_vecRewards, true);
}

/****************************************/
/****************************************/
void CCollectiveRLTransport::SocketSendRobotStats(){
  CByteArray b_vecStats;
  for(size_t i = 0; i < m_vecStats.size(); ++i){
    appendFloatIEEE754(b_vecStats, m_vecStats[i]);
  }
  socket.SendMsg(b_vecStats, false);
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::SocketGetActions(CByteArray& b) {
   /* Receive the message */
   socket.RecvMsg(b);
   const float* pfPtr = reinterpret_cast<const float*>(b.ToCArray());
   m_vecActions = std::vector<float>(pfPtr, pfPtr + m_unNumRobots*m_unNumActions);
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::SocketGetAck() {
   DEBUG_FUNCTION_ENTER;
   /* Receive the message */
   CByteArray ack;
   socket.RecvMsg(ack);
   if(ack[0] != 111 || ack[1] != 107) {
      THROW_ARGOSEXCEPTION("Cannot receive acknowledgment from PyTorch");
   }
   DEBUG_FUNCTION_EXIT;
}

/****************************************/
/****************************************/

CByteArray& appendFloatIEEE754(CByteArray& b, float f){
  b << (*reinterpret_cast<UInt32*>(&f));
  return b;
}

/****************************************/
/****************************************/

CByteArray& getFloatIEEE754(CByteArray& b, float& f){
  UInt32 buffer;
  b >> buffer;
  f = *reinterpret_cast<float*>(&buffer);
  return b;
}

REGISTER_LOOP_FUNCTIONS(CCollectiveRLTransport, "collective_rl_transport");
