#include "collectiveRlTransport.h"
#include <buzz/buzzvm.h>
#include <zmq.h>

using namespace argos;

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
   m_unNumObs(7),
   m_unNumActions(2),
   m_ptZMQContext(nullptr),
   m_ptZMQSocket(nullptr) {
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Init(TConfigurationNode& t_tree) {
   try {
      /* Parse XML tree */
      GetNodeAttribute(t_tree, "data_file",       m_strOutFile);
      GetNodeAttribute(t_tree, "num_robots",      m_unNumRobots);
      GetNodeAttribute(t_tree, "goal",            m_cGoal);
      GetNodeAttribute(t_tree, "threshold",       m_fThreshold);
      GetNodeAttribute(t_tree, "num_episodes",    m_unNumEpisodes);
      GetNodeAttribute(t_tree, "episode_time",    m_unEpisodeTime);
      GetNodeAttribute(t_tree, "time_out_reward", m_fTimeOutReward);
      GetNodeAttribute(t_tree, "threshold_freq", m_unDecThresholdTime);
      GetNodeAttribute(t_tree, "threshold_dec", m_fDecThreshold);
      GetNodeAttribute(t_tree, "min_threshold", m_fMinThreshold);
      GetNodeAttribute(t_tree, "goal_reward",     m_fGoalReward);
      std::string strPyTorchURL;
      GetNodeAttribute(t_tree, "pytorch_url",     strPyTorchURL);

      /*
       * Connect to PyTorch
       */
      /* Create context */
      m_ptZMQContext = zmq_ctx_new();
      if(m_ptZMQContext == nullptr) {
         THROW_ARGOSEXCEPTION("Cannot create ZeroMQ context: " << zmq_strerror(errno));
      }
      /* Create context */
      m_ptZMQSocket = zmq_socket(m_ptZMQContext, ZMQ_REQ);
      if(m_ptZMQSocket == nullptr) {
         THROW_ARGOSEXCEPTION("Cannot create ZeroMQ socket: " << zmq_strerror(errno));
      }
      /*
       * This call returns success even when the server is down.
       * Usually a failed connection is detected when the first data is sent.
       * This weird behavior is a design choice of ZeroMQ.
       */
      if(zmq_connect(m_ptZMQSocket, strPyTorchURL.c_str()) != 0) {
         THROW_ARGOSEXCEPTION("Cannot connect to " << strPyTorchURL << ": " << zmq_strerror(errno));
      }
      /* Send parameters */
      ZMQSendParams();
      LOG << "[INFO] Connection to PyTorch server " << strPyTorchURL << " successful" << std::endl;
      /* Initialize episode-related variables */
      m_bReachedGoal = false;
      m_unEpisodeCounter = 0;
      m_unEpisodeTicksLeft = m_unEpisodeTime;
      /* Create structures for observations, reward, and actions */
      m_vecObs.resize(m_unNumObs * m_unNumRobots, 0.0);
      m_vecRewards.resize(m_unNumRobots, 0.0);
      m_vecActions.resize(m_unNumActions * m_unNumRobots, 0.0);
      /* Create a new RNG */
      m_pcRNG = CRandom::CreateRNG("argos");
      /* Create and place stuff */
      CreateEntities();
      PlaceEntities(0);
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
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Reset() {
   PlaceEntities(m_unEpisodeCounter);
   m_bReachedGoal = false;
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Destroy() {
   /* Disconnect and get rid of the ZeroMQ socket */
   if(m_ptZMQSocket) zmq_close(m_ptZMQSocket);
   /* Get rid of the ZeroMQ context */
   if(m_ptZMQContext) zmq_ctx_destroy(m_ptZMQContext);
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
      /*DEBUG("[Ex] [t=%u] [R=%u] A = %f,%f\n",
            CSimulator::GetInstance().GetSpace().GetSimulationClock(),
            t_vm->robot,
            LIncrease[t_vm->robot],
            RIncrease[t_vm->robot]);*/
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
      m_vecRewards[i] = fReward;
   }

}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PreStep() {
   GetObservations(EPISODE_RUNNING);
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
   ZMQSendEpisodeState(EPISODE_RUNNING);
   ZMQSendObservations();
   ZMQSendRewards();
   /* Get actions from PyTorch */
   ZMQGetActions();
   /*for(size_t i = 0; i < m_unNumRobots; ++i) {
      float* pfAction = &m_vecActions[0] + i * m_unNumActions;
      DEBUG("[E%u] [t=%u] [R=%zu] RAW A = %f,%f\n",
            m_unEpisodeCounter,
            GetSpace().GetSimulationClock(),
            i,
            pfAction[0],
            pfAction[1]);
   }*/
   std::vector<Real> vecLIncrease(m_unNumRobots);
   std::vector<Real> vecRIncrease(m_unNumRobots);
   for(size_t i = 0; i < m_vecRobots.size(); ++i) {
      float* pfAction = &m_vecActions[0] + i * m_unNumActions;
      vecLIncrease[i] = pfAction[0];
      vecRIncrease[i] = pfAction[1];
   }
   BuzzForeachVM(PutIncreases(vecLIncrease, vecRIncrease));
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
   ZMQSendTermination();
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
      LOG << "Episode " << m_unEpisodeCounter + 1 << " is done" << std::endl;
      /* check to see if we need to decrease the threshold */
      if((m_unEpisodeCounter+1) % m_unDecThresholdTime == 0){
        if(m_fThreshold > m_fMinThreshold){
          m_fThreshold = m_fThreshold - m_fDecThreshold;
          LOG << "Updating Threshold to:"<< m_fThreshold <<std::endl;
        }
      }

      eState = m_bReachedGoal ? EPISODE_SUCCESS : EPISODE_TIMEOUT;
      GetObservations(eState);
      /*for(size_t i = 0; i < m_unNumRobots; ++i) {
         for(size_t j = 0; j < m_unNumObs; ++j) {
            DEBUG("[E%u] [t=%u] %s = %f\n", m_unEpisodeCounter, GetSpace().GetSimulationClock(), OBS_DESCRIPTIONS[j].c_str(), m_vecObs[i * m_unNumObs + j]);
         }
      }
      DEBUG("\n");*/
      /* Send observations to PyTorch */
      ZMQSendEpisodeState(eState);
      ZMQSendObservations();
      ZMQSendRewards();
      ZMQGetAck();
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

void CCollectiveRLTransport::ZMQSendEpisodeState(EEpisodeState e_state) {
   unsigned char punDone[3] =
      {
         0,                            // experiment not done
         (e_state != EPISODE_RUNNING), // episode done?
         m_bReachedGoal,               // reached goal?
      };
   if(zmq_send(
         m_ptZMQSocket,             // the socket
         &punDone,                   // data pointer
         sizeof(unsigned char) * 3, // data size in bytes
         ZMQ_SNDMORE)               // another message will follow (observations)
      < 0) {                        // >= 0 means success
      THROW_ARGOSEXCEPTION("Cannot send episode state to PyTorch: " << zmq_strerror(errno));
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::ZMQSendTermination() {
   unsigned char punDone[3] =
      {
         1, // experiment done
         1, // episode done
         0, // No Reward
      };
   if(zmq_send(
         m_ptZMQSocket,             // the socket
         &punDone,                  // data pointer
         sizeof(unsigned char) * 3, // data size in bytes
         0)                         // no more messages
      < 0) {                        // >= 0 means success
      THROW_ARGOSEXCEPTION("Cannot send episode state to PyTorch: " << zmq_strerror(errno));
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::ZMQSendParams() {
   /* Make the parameter buffer */
   std::vector<unsigned int> vecParams;
   vecParams.push_back(m_unNumRobots);
   vecParams.push_back(m_unNumObs);
   vecParams.push_back(m_unNumActions);
   /*DEBUG("m_unNumRobots  = %u\n", m_unNumRobots);
   DEBUG("m_unNumObs     = %u\n", m_unNumObs);
   DEBUG("m_unNumActions = %u\n", m_unNumActions);*/
   /* Send the parameters */
   if(zmq_send(
         m_ptZMQSocket,                           // the socket
         &vecParams[0],                           // data pointer
         sizeof(unsigned int) * vecParams.size(), // data size in bytes
         0)                                       // no special flags
      < 0) {                                      // >= 0 means success
      THROW_ARGOSEXCEPTION("Cannot send parameters to PyTorch: " << zmq_strerror(errno));
   }
   /* Wait for acknowledgment */
   ZMQGetAck();
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::ZMQSendObservations() {
   if(zmq_send_const(
         m_ptZMQSocket,                    // the socket
         const_cast<float*>(&m_vecObs[0]), // data pointer
         sizeof(float) * m_vecObs.size(),  // data size in bytes
         ZMQ_SNDMORE)                      // another message will follow (rewards)
      < 0) {                               // >= 0 means success
      THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
   }
}

/****************************************/
/****************************************/
void CCollectiveRLTransport::ZMQSendRewards(){

  if (zmq_send_const(
        m_ptZMQSocket,                        // the socket
        const_cast<float*>(&m_vecRewards[0]), // data pointer
        sizeof(float)*m_vecRewards.size(),    //data size in bytes
        0)                                    // no special flags
     < 0) {                                   // >= means success
       THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
   }
}


/****************************************/
/****************************************/

void CCollectiveRLTransport::ZMQGetActions() {
   /* Receive the message */
   if(zmq_recv(
         m_ptZMQSocket,                       // the socket
         &m_vecActions[0],                    // data buffer
         sizeof(float) * m_vecActions.size(), // data size in bytes
         0)                                   // no special flags
      < 0) {                                  // >= 0 means success
      THROW_ARGOSEXCEPTION("Cannot receive data from PyTorch: " << zmq_strerror(errno));
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::ZMQGetAck() {
   DEBUG_FUNCTION_ENTER;
   char pchAck[4];
   /* Receive the message */
   if(zmq_recv(
         m_ptZMQSocket, // the socket
         &pchAck,       // data buffer
         4,             // data size in bytes
         0)             // no special flags
      < 0) {            // >= 0 means success
      THROW_ARGOSEXCEPTION("Cannot receive acknowledgment from PyTorch: " << zmq_strerror(errno));
   }
   DEBUG_FUNCTION_EXIT;
}

/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(CCollectiveRLTransport, "collective_rl_transport");
