#include "cmPrediction.h"
#include <buzz/buzzvm.h>
#include <zmq.h>
#include <cmath>

using namespace argos;

/****************************************/
/****************************************/

static const std::string KIV_CONTROLLER = "kivc";
static const Real CYLINDER_RADIUS           = 0.12;  // m
static const Real CYLINDER_HEIGHT           = 0.25; // m
static const Real CYLINDER_MASS             = 100;  // kg
static const Real OBSTACLE_RADIUS           = 0.15;  // m
static const Real OBSTACLE_HEIGHT           = 0.5;  // m
static const Real OBSTACLE_MASS             = 100;  // kg
static const Real KHEPERAIV_RADIUS          = 0.09855f; // m
static const Real CYLINDER_OFFSET           = 0.0;
static const Real ROBOT_CYLINDER_DISTANCE   = CYLINDER_RADIUS + KHEPERAIV_RADIUS + CYLINDER_OFFSET;
// Adding an offset just makes the robots start slightly farther away
static const Real CYLINDER_PLACEMENT_RADIUS = ROBOT_CYLINDER_DISTANCE + KHEPERAIV_RADIUS;

static const std::string OBS_DESCRIPTIONS[] = {
   "parallel_force",
   "perpendicular_force",
   "robot2goal_dist",
   "robot2goal_angle",
   "robot2cylinder_dist",
   "robot2cylinder_angle",
   "robot_direction_from_wanted",
   "robot_direction"
};

static const std::string ACTIONS_DESCRIPTIONS[] = {
   "lwheel",
   "rwheel"
};

/****************************************/
/****************************************/

CCMPrediction::CCMPrediction() :
   m_unNumObs(10+8), // PF, PLF, LW, RW, SizeCyl, VecCylAnch, RobDirFrmWntdDir,RobDir, xEstimation, yEstimation, 8 proximity sensors
   m_unNumActions(3),
   m_ptZMQContext(nullptr),
   m_ptZMQSocket(nullptr) {
}

/****************************************/
/****************************************/

void CCMPrediction::Init(TConfigurationNode& t_tree) {
   try {
      /* Parse XML tree */
      LOG<<"Initiating"<<std::endl;
      GetNodeAttribute(t_tree, "data_file",       m_strOutFile);
      GetNodeAttribute(t_tree, "num_robots",      m_unNumRobots);
      GetNodeAttribute(t_tree, "threshold",       m_fThreshold);
      GetNodeAttribute(t_tree, "num_episodes",    m_unNumEpisodes);
      GetNodeAttribute(t_tree, "episode_time",    m_unEpisodeTime);
      GetNodeAttribute(t_tree, "time_out_reward", m_fTimeOutReward);
      GetNodeAttribute(t_tree, "threshold_freq", m_unDecThresholdTime);
      GetNodeAttribute(t_tree, "threshold_dec", m_fDecThreshold);
      GetNodeAttribute(t_tree, "min_threshold", m_fMinThreshold);
      GetNodeAttribute(t_tree, "prediction_reward", m_fPredictionReward);
      GetNodeAttribute(t_tree, "ticks_per_direction_switch", m_unTicksPerDuration);
      std::string strPyTorchURL;
      GetNodeAttribute(t_tree, "pytorch_url",     strPyTorchURL);
      GetNodeAttribute(t_tree, "alphabet_size", m_unAlphabetSize);
      GetNodeAttribute(t_tree, "proximity_range", m_fProximityRange);
      GetNodeAttribute(t_tree, "use_base_model", m_unBaseModel);
      GetNodeAttributeOrDefault(t_tree, "simulate_robots", m_bSimulateRobots, true);
      GetNodeAttributeOrDefault(t_tree, "simulate_object_mass_offset", m_bSimulateObjectMO, false);

      GetNodeAttributeOrDefault(t_tree, "robots_used", m_strRobotsUsed, m_strRobotsUsed); // Defaults are these because these are the robots we initially tested on

      m_fKheperaIVAxelLength = 0.14; // m
      m_fKheperaIVWheelRadius = 0.029112741; // m

      LOG << "Attributes taken from argos file" << std::endl;

      /* Stats to be sent to Data: */
      m_unNumStats = 4;
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
      if(zmq_connect(m_ptZMQSocket, strPyTorchURL.c_str()) != 0) {
         THROW_ARGOSEXCEPTION("Cannot connect to " << strPyTorchURL << ": " << zmq_strerror(errno));
      }
      /* Send parameters */
      ZMQSendParams(); // TODO : Setup the receiver on the python end for init
      LOG << "[INFO] Connection to PyTorch server " << strPyTorchURL << " successful" << std::endl;
      /* Initialize episode-related variables
       * This weird behavior is a design choice of ZeroMQ.
       */
      m_unEpisodeCounter = 0;
      m_unEpisodeTicksLeft = m_unEpisodeTime;
      /* Create structures for observations, reward, and actions */
      m_vecObs.resize(m_unNumObs * m_unNumRobots, 0.0);
      m_vecFailures.resize(m_unNumRobots, 0);
      m_vecRewards.resize(m_unNumRobots, 0.0);
      m_vecStats.resize(m_unNumRobots * m_unNumStats, 0.0);
      m_vecRobotStats.resize(m_unNumRobots*6, 0.0);

      m_xOffsetFromRobot.resize(m_unNumRobots);
      m_yOffsetFromRobot.resize(m_unNumRobots);

      m_vecActions.resize(m_unNumActions * m_unNumRobots, 0.0);
      /* Create a new RNG */
      LOG<<"[INFO] Creating RNG for Training"<<std::endl;
      m_pcRNG = CRandom::CreateRNG("argos");
      /* Create and place stuff */
      if(m_bSimulateRobots){
        SimulateRobots();
        LOG<<"Robots Simulated"<<std::endl;
      } else {
        SimulateRobots();
        LOG<<"Robots taken from field"<<std::endl;
      }
      //Print first failure set
      for(size_t i = 0; i < m_unNumRobots; i++){
        LOG << m_vecRobotFailures[m_unEpisodeCounter][i]<<" ";
      }
      LOG<<std::endl;

      PlaceRobots(0);

      Intended_Dir = CRadians::ZERO;

      CBuzzLoopFunctions::Init(t_tree);
//    }
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("while initializing the loop functions", ex);
   }
}

/****************************************/
/****************************************/

void CCMPrediction::SimulateRobots() {
   if(m_bSimulateObjectMO){
       // TODO Use Julian's code here
   } else {
       // TODO : Change this to a square to allow robot to always be perfectly perpendicular
//       m_pcBox = new CBoxEntity("Box_1",
//                                CVector3(),
//                                CQuaternion(),
//                                true,
//                                CVector3(CYLINDER_RADIUS,CYLINDER_RADIUS,CYLINDER_HEIGHT),
//                                CYLINDER_MASS);
//       AddEntity(m_pcBox); // TODO : Make this code work

       /* Create the cylinder */
       m_pcCylinder = new CCylinderEntity(
          "Cylinder_1",
          CVector3(),
          CQuaternion(),
          true,
          CYLINDER_RADIUS,
          CYLINDER_HEIGHT,
          CYLINDER_MASS);
       AddEntity(*m_pcCylinder);
   }
   /* Create robots */
   CRadians cSlice = CRadians::TWO_PI / m_unNumRobots;
   std::ostringstream cKIVId;
   CKheperaIVEntity* pcKIV;
   CVector3 cPos;
   /* Create the robots in simulation*/
   std::vector<int> vect;
   std::stringstream ss(m_strRobotsUsed);

   for (int i; ss >> i;) {
       vect.push_back(i);
       if (ss.peek() == ',')
           ss.ignore();
   }

   for(size_t i = 0; i < m_unNumRobots; ++i) {
      cKIVId.str("");
      cKIVId << "Khepera_" << vect[i];
      LOG<<"Adding "<<cKIVId.str()<<" to Environment"<<std::endl;
      cPos.FromSphericalCoords(ROBOT_CYLINDER_DISTANCE,
                               CRadians::PI_OVER_TWO,
                               i * cSlice);
      cPos.SetZ(0.0);
      pcKIV = new CKheperaIVEntity(
         cKIVId.str(),
         KIV_CONTROLLER,
         cPos,
         CQuaternion(-cSlice, CVector3::Z)
         );
      m_vecRobots.push_back(pcKIV);

      m_xOffsetFromRobot.push_back(0.0);
      m_yOffsetFromRobot.push_back(0.0);

      /** Need to chage the range of the proximity sensor
          If m_fProximityRange = 0 then the sensor range will
          stay at default */
      AddEntity(*pcKIV);
      if(m_fProximityRange > 0.0){
        CProximitySensorEquippedEntity& cPSEE = pcKIV->GetProximitySensorEquippedEntity();
        CProximitySensorEquippedEntity::SSensor::TList& listPS = cPSEE.GetSensors();
        for(auto itSensor = listPS.begin(); itSensor != listPS.end(); ++itSensor){
          (*itSensor)->Direction.Normalize();
          (*itSensor)->Direction *= m_fProximityRange;
        }
      }
   }

//   CRange<Real> cXCylinderRange(
//      GetSpace().GetArenaLimits().GetMin().GetX() + CYLINDER_PLACEMENT_RADIUS,
//      GetSpace().GetArenaLimits().GetMin().GetX()/2 -CYLINDER_PLACEMENT_RADIUS
//      );
//   CRange<Real> cYRange(
//      GetSpace().GetArenaLimits().GetMin().GetY() + CYLINDER_PLACEMENT_RADIUS,
//      GetSpace().GetArenaLimits().GetMax().GetY() - CYLINDER_PLACEMENT_RADIUS
//      );

   for(size_t i = 0; i < m_unNumEpisodes; ++i){
      cPos.Set(0.0,
               0.0,
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
         // TODO : Confirm this works for box

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

void CCMPrediction::PlaceRobots(UInt32 un_episode){
    /* Make sure the episode is valid */
    if(un_episode >= m_unNumEpisodes) {
       THROW_ARGOSEXCEPTION("Episode " << un_episode << " is beyond the maximum of " << m_unNumEpisodes);
    }

    if(m_bSimulateRobots){
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
       LOG<<"Placed Robots in Simulation"<<std::endl;
    } else{
        LOG<<"Taking Robots from Real"<<std::endl;
    }
}

/****************************************/
/****************************************/

void CCMPrediction::Reset() {
   for(size_t i = 0; i < m_unNumRobots; i++){
     LOG << m_vecRobotFailures[m_unEpisodeCounter][i]<<" ";
   }
   LOG<<std::endl;

   if(m_bSimulateRobots)
     PlaceRobots(m_unEpisodeCounter);

   Intended_Dir = CRadians::ZERO;



}

/****************************************/
/****************************************/

void CCMPrediction::Destroy() {
   /* Disconnect and get rid of the ZeroMQ socket */
   if(m_ptZMQSocket) zmq_close(m_ptZMQSocket);
   /* Get rid of the ZeroMQ context */
   if(m_ptZMQContext) zmq_ctx_destroy(m_ptZMQContext);
}

/****************************************/
/****************************************/

struct PutIncreases : public CBuzzLoopFunctions::COperation {

   PutIncreases(std::vector<Real>& vecWheelSpeedL,
                std::vector<Real>& vecWheelSpeedR,
                std::vector<Real>& vec_direction_to_drive,
                std::vector<UInt32>& vec_base_model) :
      LSpeed(vecWheelSpeedL),
      RSpeed(vecWheelSpeedR),
      DirectionToDrive(vec_direction_to_drive),
      BaseModel(vec_base_model) {}

   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {
      BuzzPut(t_vm, "LSpeed", static_cast<float>(LSpeed[t_vm->robot]));
      BuzzPut(t_vm, "RSpeed", static_cast<float>(RSpeed[t_vm->robot]));
      BuzzPut(t_vm, "DriveDirection", static_cast<float>(DirectionToDrive[t_vm->robot]));
      BuzzPut(t_vm, "BaseModel", static_cast<int>(BaseModel[t_vm->robot]));
      /**DEBUG("[Ex] [t=%u] [R=%u] A = %f,%f F = %u\n",
            CSimulator::GetInstance().GetSpace().GetSimulationClock(),
            t_vm->robot,
            LIncrease[t_vm->robot],
            RIncrease[t_vm->robot]);*/
   }

   std::vector<Real> LSpeed;
   std::vector<Real> RSpeed;
   std::vector<Real> DirectionToDrive;
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

struct GetForceVector : public CBuzzLoopFunctions::COperation {
   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm){
      PerpendicularForce.push_back(buzzobj_getfloat(BuzzGet(t_vm, "Perpendicular_Force")));
      ParallelForce.push_back(buzzobj_getfloat(BuzzGet(t_vm, "Perpendicular_Force")));
   }
   std::vector<float> PerpendicularForce;
   std::vector<float> ParallelForce;
};

void CCMPrediction::GetObservations(EEpisodeState e_state){
   /** Get the position and orientation of the object*/
   CVector3& cCylinderPos =
      m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position;
   CQuaternion cCylinderOrient =
      m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Orientation;
   CRadians cObjZ, cObjY, cObjX;
   cCylinderOrient.ToEulerAngles(cObjZ, cObjY, cObjX);
   /** Store object position and orientation to send to python*/
   m_vecObjStats[0] = cCylinderPos.GetX();
   m_vecObjStats[1] = cCylinderPos.GetY();
   m_vecObjStats[2] = cCylinderPos.GetZ();
   m_vecObjStats[3] = ToDegrees(cObjX).GetValue();
   m_vecObjStats[4] = ToDegrees(cObjY).GetValue();
   m_vecObjStats[5] = ToDegrees(cObjZ).GetValue();

   for(size_t i = 0; i < m_vecRobots.size(); ++i) {
      /* Get robot pose */
      CVector3& cRobotPos =
         m_vecRobots[i]->GetEmbodiedEntity().GetOriginAnchor().Position;
      CQuaternion& cRobotOrient =
         m_vecRobots[i]->GetEmbodiedEntity().GetOriginAnchor().Orientation;
      CRadians cRobotZ, cRobotY, cRobotX;
      cRobotOrient.ToEulerAngles(cRobotZ, cRobotY, cRobotX);
      /* Save Robot Positions and Orientations to send to python for MME learning (Stephen Powers)*/
      m_vecRobotStats[i*6] = cRobotPos.GetX();
      m_vecRobotStats[i*6+1] = cRobotPos.GetY();
      m_vecRobotStats[i*6+2] = cRobotPos.GetZ(); // Do we need this? Probbly NOT
      m_vecRobotStats[i*6+3] = ToDegrees(cRobotX).GetValue(); // Do we need these? Probably NOT
      m_vecRobotStats[i*6+4] = ToDegrees(cRobotY).GetValue();
      m_vecRobotStats[i*6+5] = ToDegrees(cRobotZ).GetValue();

      /* Get vector from robot to cylinder (robot-local) */
      CVector2 cVecRobot2Cylinder(
         cCylinderPos.GetX() - cRobotPos.GetX(),
         cCylinderPos.GetY() - cRobotPos.GetY());
      cVecRobot2Cylinder.Rotate(-cRobotZ);
      /* Get the direction the robot is facing in */
//      Real fRobotAngle = ToDegrees(cRobotPos.GetZAngle()).GetValue(); // The robot angle in radians

      /* Check if the robot has failed */
//      float hasFailed = 0;
//      UInt32 ticksElapsed = m_unEpisodeTime - m_unEpisodeTicksLeft;
//      if (m_vecRobotFailures[m_unEpisodeCounter][i] != -1 && m_vecRobotFailures[m_unEpisodeCounter][i] <= ticksElapsed) {
//          hasFailed = 1;
//      }


      /* Calculate reward */
      Real fReward;
      switch(e_state) {
         case EPISODE_RUNNING: {
            //fReward = -1.0 + (1.0 / (10.0 * cVecRobot2Goal.Length()));
            /* Cost of living + direction x reward for moving */
            fReward = -2;

            LOG << "Predicted Distance is " << PredictionDistance(i) << std::endl;
            fReward += m_fPredictionReward * PredictionDistance(i); // TODO : Figure out what this is returning
            break;
         }
         case EPISODE_SUCCESS: {
            fReward = 200;
            break;
         }
         case EPISODE_TIMEOUT: {
            fReward = m_fTimeOutReward;
            break;
         }
      }
      m_vecRewards[i] = fReward;


      //DEBUG("cMotion = (%f,%f)\n", cMotion.GetX(), cMotion.GetY());
      //DEBUG("cVecCylinder2Goal = (%f,%f)\n", cVecCylinder2Goal.GetX(), cVecCylinder2Goal.GetY());
      //DEBUG("fDirection = %f\n", fDirection);
      /* Get the wheel speeds*/
      GetWheelSpeeds cGWS;
      BuzzForeachVM(cGWS);
      Real fLWheel = cGWS.LWheels[i];
      Real fRWheel = cGWS.RWheels[i];
      const CVector2& cForceVector = m_vecRobots[i]->GetControllableEntity().GetController().GetSensor <CCI_KheperaIVGripperForceSensor> ("kheperaiv_gripper_force")->GetReadings();
      Real fPerpendicularForce = cForceVector.GetY(); // NOTE : These are correct function calls
      Real fParallelForce = cForceVector.GetX();
      const CQuaternion cRobotOrientation =
         m_vecRobots[i]->GetEmbodiedEntity().GetOriginAnchor().Orientation;
      CRadians cObjZ, cObjY, cObjX;
      cRobotOrientation.ToEulerAngles(cObjZ, cObjY, cObjX);
      Real fRobotOrientation = ToDegrees(cObjZ).GetValue();
      Real fTargetDirection = ToDegrees(Intended_Dir).GetValue();
      DEBUG("Perpendicular Force : %f, Parallel Force : %f", fPerpendicularForce, fParallelForce);
      /* Store the observations */
      m_vecObs[i * m_unNumObs + 0] = fPerpendicularForce;
      m_vecObs[i * m_unNumObs + 1] = fParallelForce;
      m_vecObs[i * m_unNumObs + 2] = fLWheel;
      m_vecObs[i * m_unNumObs + 3] = fRWheel;
      m_vecObs[i * m_unNumObs + 4] = cVecRobot2Cylinder.Length();
      m_vecObs[i * m_unNumObs + 5] = ToDegrees(cVecRobot2Cylinder.Angle()).GetValue();
      m_vecObs[i * m_unNumObs + 6] = fTargetDirection; // Angle to the selected direction in radians
      m_vecObs[i * m_unNumObs + 7] = fRobotOrientation;

      // Individual robot's prediction and the CM relative to the robot
      m_vecObs[i * m_unNumObs + 8] = m_xOffsetFromRobot[i];
      m_vecObs[i * m_unNumObs + 9] = m_yOffsetFromRobot[i];

      // Get the proximity sensor values
      const std::vector<argos::CCI_KheperaIVProximitySensor::SReading>& tReadings =
        m_vecRobots[i]->GetControllableEntity().GetController().GetSensor <CCI_KheperaIVProximitySensor> ("kheperaiv_proximity")->GetReadings();
      for(size_t t = 0; t < tReadings.size(); t++){
        m_vecObs[i * m_unNumObs + 8 + t] = tReadings[t].Value;
      }

   }

}

/****************************************/
/****************************************/

void CCMPrediction::PreStep() {
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
   ZMQSendEpisodeState(EPISODE_RUNNING);
   ZMQSendObservations();
   ZMQSendFailures();
   ZMQSendRewards();
   ZMQSendForceStats();
   ZMQSendRobotStats();

   ZMQSendObjectStatsFinal();

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
   std::vector<UInt32> vecBaseModel(m_unNumRobots);
   for(size_t i = 0; i < m_vecRobots.size(); ++i) {
      float* pfAction = &m_vecActions[0] + i * m_unNumActions;
      float* pfObs = &m_vecObs[0] + i * m_unNumObs;
      m_xOffsetFromRobot[i] += pfAction[0]; // Receiving the offset of x and y for each individual robot for prediction of cm robot to object
      m_xOffsetFromRobot[i] += pfAction[1];
      vecBaseModel[i] = m_unBaseModel;
   }

   std::vector<Real> vecWheelSpeedL(m_unNumRobots);
   std::vector<Real> vecWheelSpeedR(m_unNumRobots);
   std::vector<Real> vecDirectionToDrive(m_unNumRobots);
   for(size_t i = 0; i < m_vecRobots.size(); i++){
//   m_vecRobotStats[i*6 + 5] // Robot z in degrees
       Real error = ToDegrees(Intended_Dir).GetValue() - m_vecRobotStats[i*6 + 5];
       if(error < 3){
           vecWheelSpeedL[i] = std::max((error * wheel_gain), min_wheel_speed);
           vecWheelSpeedR[i] = - std::max((error * wheel_gain), min_wheel_speed);
       } else{
           vecWheelSpeedL[i] = std::max((error * wheel_gain) + 8.0, 7.0);
           vecWheelSpeedR[i] = std::max( - (error * wheel_gain) + 8.0, 7.0);
       }
   }
   BuzzForeachVM(PutIncreases(vecWheelSpeedL, vecWheelSpeedR, vecDirectionToDrive, vecBaseModel));
}

/****************************************/
/****************************************/

bool CCMPrediction::IsExperimentFinished() {
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

Real CCMPrediction::PredictionDistance(int robot_index){
    // Math to get vector of robot to gripper anchor and then robot anchor to what the predicted object center
    // of mass is, then the euclidian distance between the predicted and actual
    CVector3& cCMObject = m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position; // TODO : This is the entity's origin anchor, not center of mass, when using Julian's code, change it to be the center of mass of the object
    CVector3& cCenterRobot = m_vecRobots[robot_index]->GetEmbodiedEntity().GetOriginAnchor().Position;
    CKheperaIVGripperEntity& cGripper = m_vecRobots[robot_index]->GetGripperEquippedEntity();
    CVector3 cAnchorGripper = cGripper.GetAnchor1();
    CVector3 cPredictionModifier = CVector3(m_xOffsetFromRobot[robot_index], m_yOffsetFromRobot[robot_index], 0.0);
    CVector3 robot_grip_direction = cAnchorGripper - cCenterRobot;
    cPredictionModifier.RotateZ(cAnchorGripper.GetAngleWith(CVector3::X));
    CVector3 cPredictedCM = cAnchorGripper + cPredictionModifier;
    DEBUG("ROBOT ID %d, Modifier X : %f, Modifier Y %f \n Predicted Location X : %f, Predicted Location Y : %f, \n Actual X : %f, Actual Y : %f", robot_index, m_xOffsetFromRobot[robot_index], m_yOffsetFromRobot[robot_index], cPredictedCM.GetX(), cPredictedCM.GetY(), cCMObject.GetX(), cCMObject.GetY());
    return sqrt(pow((cPredictedCM.GetX() - cCMObject.GetX()),2) + pow((cPredictedCM.GetY() - cCMObject.GetY()),2));
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
void CCMPrediction::PostExperiment() {
   LOG<<"Closing the Server"<<std::endl;
   ZMQSendTermination();
}

/****************************************/
/****************************************/

void CCMPrediction::PostStep() {
   /* Decrement remaining time */
   --m_unEpisodeTicksLeft;
   /* Check if the cylinder reached the goal */
   m_bFoundCM = false; // TODO : Create this stop condition properly

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

      eState = m_bFoundCM ? EPISODE_SUCCESS : EPISODE_TIMEOUT;
      GetObservations(eState);
      CalculateRobotStats();
      /*for(size_t i = 0; i < m_unNumRobots; ++i) {
         for(size_t j = 0; j < m_unNumObs; ++j) {
            DEBUG("[E%u] [t=%u] %s = %f\n", m_unEpisodeCounter, GetSpace().GetSimulationClock(), OBS_DESCRIPTIONS[j].c_str(), m_vecObs[i * m_unNumObs + j]);
         }
      }
      DEBUG("\n");*/
      /* Send observations to PyTorch */
      ZMQSendEpisodeState(eState);
      ZMQSendObservations();
      ZMQSendFailures();
      ZMQSendRewards();
      ZMQSendForceStats();
      ZMQSendRobotStats();
      ZMQSendObjectStatsFinal();
      ZMQGetAck();
      /* Restart episode */
      ++m_unEpisodeCounter;
      if(m_unEpisodeCounter < m_unNumEpisodes) {
         m_unEpisodeTicksLeft = m_unEpisodeTime;
         GetSimulator().Reset();
      }

      if(m_unEpisodeCounter % m_unTicksPerDuration == 0){
          LOG << "Changing Robot Directions" << std::endl;
          Real direction_vector = Real(m_unEpisodeCounter / m_unTicksPerDuration);
          Intended_Dir = (CRadians::PI / 6.0 * direction_vector);
      }
   }
}

/****************************************/
/****************************************/

bool CCMPrediction::FoundCM(Real fXCM, Real fYCM) {
//   CVector3& cCMPrediction =
//      m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position;
//   CVector2 cCylinder2Goal(
//      m_cGoal.GetX() - cCylinderPos.GetX(),
//      m_cGoal.GetY() - cCylinderPos.GetY());
// TODO : GetCM for the cylinder and receive what our prediction is
//   return cCylinder2Goal.Length() < m_fThreshold;
   return false;
}

/****************************************/
/****************************************/

bool CCMPrediction::IsEpisodeFinished() {
   if(m_bFoundCM) {
      LOG << "CM Found within threshold" << std::endl;
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

void CCMPrediction::CalculateRobotStats(){
  for(size_t i = 0; i < m_vecRobots.size(); ++i) {
     /* Get robot pose */
     CVector3& cRobotPos =
        m_vecRobots[i]->GetEmbodiedEntity().GetOriginAnchor().Position;
     /* Get robot orientation*/
     CQuaternion& cRobotOrient = // TODO : Robot orientation is saved here, do we need it in the observations as well?
        m_vecRobots[i]->GetEmbodiedEntity().GetOriginAnchor().Orientation;
     CRadians cRobotZ, cRobotY, cRobotX;
     cRobotOrient.ToEulerAngles(cRobotZ, cRobotY, cRobotX);
     /* Get the wheel speeds*/
     GetWheelSpeeds cGWS;
     BuzzForeachVM(cGWS);
     Real fLWheel = cGWS.LWheels[i];
     Real fRWheel = cGWS.RWheels[i];

     /*Calculate force vector*/
     Real deltaX = (m_fKheperaIVWheelRadius/2.0) * (fLWheel + fRWheel) * Cos(cRobotZ);
     Real deltaY = (m_fKheperaIVWheelRadius/2.0) * (fLWheel + fRWheel) * Sin(cRobotZ);

     Real magnitude = Sqrt((cRobotPos.GetX() - deltaX)*(cRobotPos.GetX() - deltaX)
                            + (cRobotPos.GetY() - deltaY)*(cRobotPos.GetY() - deltaY));

     m_vecStats[i * m_unNumStats + 0] = magnitude;
     m_vecStats[i * m_unNumStats + 1] = ToDegrees(cRobotZ).GetValue();
     m_vecStats[i * m_unNumStats + 2] = deltaX;
     m_vecStats[i * m_unNumStats + 3] = deltaY;
   }
}



/****************************************/
/****************************************/

void CCMPrediction::ZMQSendEpisodeState(EEpisodeState e_state) {
   unsigned char punDone[3] =
      {
         0,                            // experiment not done
         (e_state != EPISODE_RUNNING), // episode done?
         m_bFoundCM,               // reached goal?
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

void CCMPrediction::ZMQSendTermination() {
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

void CCMPrediction::ZMQSendParams() {
   /* Make the parameter buffer */ // TODO update ZMQ Send Params
   std::vector<float> vecParams;
   vecParams.push_back(m_unNumRobots);
   vecParams.push_back(m_unNumObs);
   vecParams.push_back(m_unNumActions);
   vecParams.push_back(m_unNumStats);
   vecParams.push_back(m_unAlphabetSize);



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

void CCMPrediction::ZMQSendObservations() {
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

void CCMPrediction::ZMQSendFailures() {
   /*LOG<<"[DEBUG] Sending Failures ";
   for(size_t i = 0; i < m_unNumRobots; ++i){
     LOG<<m_vecFailures[i]<<" ";
   }
   LOG<<std::endl;*/
   if(zmq_send_const(
         m_ptZMQSocket,                    // the socket
         const_cast<int*>(&m_vecFailures[0]), // data pointer
         sizeof(int) * m_vecFailures.size(),  // data size in bytes
         ZMQ_SNDMORE)                      // another message will follow (rewards)
      < 0) {                               // >= 0 means success
      THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
   }
}

/****************************************/
/****************************************/
void CCMPrediction::ZMQSendRewards(){

  if (zmq_send_const(
        m_ptZMQSocket,                        // the socket
        const_cast<float*>(&m_vecRewards[0]), // data pointer
        sizeof(float)*m_vecRewards.size(),    //data size in bytes
        ZMQ_SNDMORE)                          // another message will follow (Stats)
     < 0) {                                   // >= means success
       THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
   }
}

/****************************************/
/****************************************/
void CCMPrediction::ZMQSendForceStats(){
  if (zmq_send_const(
    m_ptZMQSocket,                        // The socket
    const_cast <float*>(&m_vecStats[0]),  // data pointer
    sizeof(float)*m_vecStats.size(),      // data size in bytes
    ZMQ_SNDMORE)                         // another message will follow (object stats)
  < 0) {                                  // >= 0 means success
    THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
  }
}

/****************************************/
/****************************************/
void CCMPrediction::ZMQSendObjectStats(){
  if (zmq_send_const(
    m_ptZMQSocket,                        // The socket
    const_cast <float*>(&m_vecObjStats[0]),  // data pointer
    sizeof(float)*m_vecObjStats.size(),      // data size in bytes
    ZMQ_SNDMORE)                                    //more to come
  < 0) {                                  // >= 0 means success
    THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
  }
}

/****************************************/
/****************************************/
void CCMPrediction::ZMQSendRobotStats(){
  if (zmq_send_const(
    m_ptZMQSocket,                        // The socket
    const_cast <float*>(&m_vecRobotStats[0]),  // data pointer
    sizeof(float)*m_vecRobotStats.size(),      // data size in bytes
    ZMQ_SNDMORE)                                    //more to come
  < 0) {                                  // >= 0 means success
    THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
  }
}

/****************************************/
/****************************************/
void CCMPrediction::ZMQSendObjectStatsFinal(){
  if (zmq_send_const(
    m_ptZMQSocket,                        // The socket
    const_cast <float*>(&m_vecObjStats[0]),  // data pointer
    sizeof(float)*m_vecObjStats.size(),      // data size in bytes
    0)                                    // final message
  < 0) {                                  // >= 0 means success
    THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
  }
}

/****************************************/
/****************************************/

void CCMPrediction::ZMQGetActions() {
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

void CCMPrediction::ZMQGetAck() {
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

REGISTER_LOOP_FUNCTIONS(CCMPrediction, "cm_prediction");
