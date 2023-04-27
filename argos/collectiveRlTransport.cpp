#include "collectiveRlTransport.h"
#include <buzz/buzzvm.h>
#include <zmq.h>
#include <cmath>

using namespace argos;

/****************************************/
/****************************************/

//static const std::string FB_CONTROLLER = "fgc";
static const std::string KIV_CONTROLLER = "kivc"; // Chandler : swapped FB_CONTROLLER with KIV_CONTROLLER, unsure if correct name
// TODO : fihure out what goes here for khepera
static const Real WALL_THICKNESS            = 0.2;  // m
static const Real CYLINDER_RADIUS           = 0.09855;  // m Previously 0.5 for footbot
static const Real CYLINDER_HEIGHT           = 0.25; // m
static const Real CYLINDER_MASS             = 100;  // kg
static const Real OBSTACLE_RADIUS           = 0.5;  // m
static const Real OBSTACLE_HEIGHT           = 0.5;  // m
static const Real OBSTACLE_MASS             = 100;  // kg
// Chandler : Changed FOOTBOT_RADIUS to KHEPERAIV_RADIUS
//static const Real FOOTBOT_RADIUS            = 0.085036758f; // m
static const Real KHEPERAIV_RADIUS          = 0.09855f; // m // TODO : Get the real radius WITH gripper
static const Real CYLINDER_OFFSET           = 0.0;
static const Real ROBOT_CYLINDER_DISTANCE   = CYLINDER_RADIUS + KHEPERAIV_RADIUS + CYLINDER_OFFSET;  // m Previously 0.6
// Adding an offset just makes the robots start slightly farther away
static const Real CYLINDER_PLACEMENT_RADIUS = WALL_THICKNESS + ROBOT_CYLINDER_DISTANCE + KHEPERAIV_RADIUS;

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
   m_unNumObs(7+8), // Was m_unNumObs(7+24) but kheperaiv has 8 proximity sensors, not 24
   m_unNumActions(3),
   m_ptZMQContext(nullptr),
   m_ptZMQSocket(nullptr) {
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
      std::string strPyTorchURL;
      GetNodeAttribute(t_tree, "pytorch_url",     strPyTorchURL);
      GetNodeAttribute(t_tree, "alphabet_size", m_unAlphabetSize);
      GetNodeAttribute(t_tree, "proximity_range", m_fProximityRange);
      GetNodeAttribute(t_tree, "num_obstacles", m_unNumObstacles);
      GetNodeAttribute(t_tree, "use_gate", m_unUseGate);
      GetNodeAttribute(t_tree, "gate_curriculum", m_unGateCurriculum);
      GetNodeAttribute(t_tree, "gate_update_frequency", m_unGateUpdateFrequency);
      GetNodeAttribute(t_tree, "gate_update_amount", m_fGateUpdate);
      GetNodeAttribute(t_tree, "gate_minimum", m_fGateMinimum);
      GetNodeAttribute(t_tree, "use_base_model", m_unBaseModel);

      /* Footbot dynamic equation parameters*/
      // TODO : Grap the actual values for khepera
      // Chandler : Changed m_fFootbotAxelLength to m_fKheperaIVAxelLength
      m_fKheperaIVAxelLength = 0.14; // m
      // Chandler : Changed m_fFootbotWheelRadius to m_fKheperaIVWheelRadius and all instances of its use
      m_fKheperaIVWheelRadius = 0.029112741; // m

      /* Stats to be sent to Data: Force vector (direction and magnitude) for every robot*/
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
      m_vecFailures.resize(m_unNumRobots, 0);
      m_vecRewards.resize(m_unNumRobots, 0.0);
      m_vecStats.resize(m_unNumRobots * m_unNumStats, 0.0);
      m_vecRobotStats.resize(m_unNumRobots*6, 0.0);
      m_vecObjStats.resize(7, 0.0);
      m_vecGateStats.resize(4, 0.0);
      if (m_unNumObstacles > 0){
          m_vecObstacleStats.resize(m_unNumObstacles*2, 0.0);
      }
      m_vecActions.resize(m_unNumActions * m_unNumRobots, 0.0);
      /* Create a new RNG */
      LOG<<"[INFO] Creating RNG for Training"<<std::endl;
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
   std::ostringstream cKIVId; // Chandler : changed cFBId to cKIVId
//   CFootBotEntity* pcFB; // Chandler : Changed all instances of CFootBotEntity and pcFB with Khepera and pcKIV
   CKheperaIVEntity* pcKIV;
   CVector3 cPos;
   for(size_t i = 0; i < m_unNumRobots; ++i) {
      cKIVId.str("");
      cKIVId << "kiv" << i; // Changed "fb" to kiv
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
   for(size_t i = 0; i < m_unNumEpisodes; ++i){
     m_vecObstaclePos.push_back(std::vector<CVector3>());
     for(size_t j = 0; j < m_unNumObstacles; j++){
       cPos.Set(m_pcRNG->Uniform(cXObstacleRange),
                m_pcRNG->Uniform(cYRange),
                0.0);
       m_vecObstaclePos.back().push_back(cPos);
     }
   }
   /**
   Generate the gate model obstacle from
   two box_entities. The gate should be in
   a different position every episode which
   means the boxes have to change shape every
   episode.
   */
   if(m_unUseGate){
     /** Create two box entities and add them to the environment*/
     std::ostringstream bCID;
     CBoxEntity* pcB;
     for(size_t i = 0; i < 2; ++i){
       bCID.str("");
       bCID << "b" << i;
       pcB = new CBoxEntity(
          bCID.str(),
          CVector3(),
          CQuaternion(),
          false,
          CVector3(),
          OBSTACLE_MASS);
       m_vecGateWalls.push_back(pcB);
       AddEntity(*pcB);
     }
     Real offset ;
     if(m_unGateCurriculum == 1){
       offset = GetSpace().GetArenaLimits().GetMax().GetY();
     }
     else{
       offset = (m_fGateMinimum/2.0);
     }
     UInt32 update_offset_flag = 1;
     CRange<Real> cXWallRange(
        GetSpace().GetArenaLimits().GetMin().GetX()/2,
        0
        );
     for(size_t i = 0; i < m_unNumEpisodes; ++i){
       if(update_offset_flag == 1){
         /** Dont update on episode 0*/
         if((i+1)%m_unGateUpdateFrequency == 0){
           offset = offset - m_fGateUpdate;
           std::cout<<"Updating gap distance to "<<offset*2<<" at episode "<<i<<std::endl;
           if(offset <= (m_fGateMinimum/2.0)){
             offset = (m_fGateMinimum/2.0);
             std::cout<<"Reached final gap distance of "<<offset*2<<" at episode "<<i<<std::endl;
             update_offset_flag = 0;
             }
           }
         }
         m_vecOffset.push_back(offset);
         CRange<Real> cYRange(
            GetSpace().GetArenaLimits().GetMin().GetY() + offset,
            GetSpace().GetArenaLimits().GetMax().GetY() - offset
            );
         m_vecGateWallPos.push_back(std::vector<CVector3>());
         m_vecGateWallSize.push_back(std::vector<CVector3>());
         /** Generate positions and sizes for the box entitites */
         Real YPos = m_pcRNG->Uniform(cYRange);
         Real XPos = m_pcRNG->Uniform(cXWallRange);
         /** set position*/
         cPos.Set(XPos,
                  GetSpace().GetArenaLimits().GetMin().GetY() + (abs(GetSpace().GetArenaLimits().GetMin().GetY() - (YPos - offset)))/2,
                  0.0);
         m_vecGateWallPos.back().push_back(cPos);
         cPos.Set(XPos,
                  GetSpace().GetArenaLimits().GetMax().GetY() - (abs(GetSpace().GetArenaLimits().GetMax().GetY() - (YPos + offset)))/2,
                  0.0);
         m_vecGateWallPos.back().push_back(cPos);
         /** Set Size */
         cPos.Set(0.5,
                  abs(GetSpace().GetArenaLimits().GetMin().GetY() - (YPos - offset)),
                  0.5);
         m_vecGateWallSize.back().push_back(cPos);
         cPos.Set(0.5,
                  abs(GetSpace().GetArenaLimits().GetMax().GetY() - (YPos + offset)),
                  0.5);
         m_vecGateWallSize.back().push_back(cPos);
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
   if(m_unUseGate == 1){
     /** Move the Walls */
     MoveEntity(m_vecGateWalls[0]->GetEmbodiedEntity(),
                m_vecGateWallPos[un_episode][0],
                CQuaternion(),
                false,
                true);
     MoveEntity(m_vecGateWalls[1]->GetEmbodiedEntity(),
                m_vecGateWallPos[un_episode][1],
                CQuaternion(),
                false,
                true);

      /** Resize the walls */
      m_vecGateWalls[0]->Resize(m_vecGateWallSize[un_episode][0]);
      m_vecGateWalls[1]->Resize(m_vecGateWallSize[un_episode][1]);
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
   if(m_unUseGate==1){
     LOG<<"[INFO] Current Offset: "<<m_vecOffset[m_unEpisodeCounter]<<std::endl;
   }
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
   if(m_unUseGate == 1){
     m_vecGateStats[0] = m_vecGateWallPos[m_unEpisodeCounter][0].GetX();
     m_vecGateStats[1] = m_vecGateWallSize[m_unEpisodeCounter][0].GetY();
     m_vecGateStats[2] = m_vecGateWallPos[m_unEpisodeCounter][1].GetX();
     m_vecGateStats[3] = m_vecGateWallSize[m_unEpisodeCounter][1].GetY();
   }
   else if(m_unNumObstacles > 0){
     for(size_t i = 0; i < m_unNumObstacles; ++i){
       m_vecObstacleStats[i*2] = m_vecObstaclePos[m_unEpisodeCounter][i].GetX();
       m_vecObstacleStats[i*2+1] = m_vecObstaclePos[m_unEpisodeCounter][i].GetY();
     }
   }

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
      m_vecRobotStats[i*6+2] = cRobotPos.GetZ();
      m_vecRobotStats[i*6+3] = ToDegrees(cRobotX).GetValue();
      m_vecRobotStats[i*6+4] = ToDegrees(cRobotY).GetValue();
      m_vecRobotStats[i*6+5] = ToDegrees(cRobotZ).GetValue();

      /* Get vector from robot to goal (robot-local) */
      CVector2 cVecRobot2Goal(
         m_cGoal.GetX() - cRobotPos.GetX(),
         m_cGoal.GetY() - cRobotPos.GetY());
      cVecRobot2Goal.Rotate(-cRobotZ);
      /* Get vector from robot to cylinder (robot-local) */
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
      /* Adding cylinder angle to goal for MME*/
      m_vecObjStats[6] = ToDegrees(cVecCylinder2Goal.Angle()).GetValue();
      // Get the proximity sensor values
      // Chandler : changed CCI_FootBotProximitySensor to CCI_KheperaIVProximitySensor
      const std::vector<argos::CCI_KheperaIVProximitySensor::SReading>& tReadings =
        m_vecRobots[i]->GetControllableEntity().GetController().GetSensor <CCI_KheperaIVProximitySensor> ("kheperaiv_proximity")->GetReadings(); // Chandler : Edited footbot_proximity to kheperaiv_proximity
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
   ZMQSendEpisodeState(EPISODE_RUNNING);
   ZMQSendObservations();
   ZMQSendFailures();
   ZMQSendRewards();
   ZMQSendForceStats();
   ZMQSendRobotStats();
   if(m_unNumObstacles> 0){
     ZMQSendObjectStats();
     ZMQSendObstacleStats();

   }
   else if(m_unUseGate == 1){
     ZMQSendObjectStats();
     ZMQSendGateStats();

   }
   else{
     ZMQSendObjectStatsFinal();
   }
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
      ZMQSendEpisodeState(eState);
      ZMQSendObservations();
      ZMQSendFailures();
      ZMQSendRewards();
      ZMQSendForceStats();
      ZMQSendRobotStats();
      if(m_unNumObstacles> 0){
        ZMQSendObjectStats();
        ZMQSendObstacleStats();
      }
      else if(m_unUseGate == 1){
        ZMQSendObjectStats();
        ZMQSendGateStats();
      }
      else{
        ZMQSendObjectStatsFinal();
      }
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
   std::vector<float> vecParams;
   vecParams.push_back(m_unNumRobots);
   vecParams.push_back(m_unNumObstacles);
   vecParams.push_back(m_unNumObs);
   vecParams.push_back(m_unNumActions);
   vecParams.push_back(m_unNumStats);
   vecParams.push_back(m_unAlphabetSize);
   vecParams.push_back(m_unUseGate);
   /* Calculate normalizing constants*/
   float maxY = GetSpace().GetArenaLimits().GetMax().GetY();
   float minY = GetSpace().GetArenaLimits().GetMin().GetY();
   float maxX = GetSpace().GetArenaLimits().GetMax().GetX();
   float minX = GetSpace().GetArenaLimits().GetMin().GetX();
   float goalX = m_cGoal.GetX();
   float goalY = m_cGoal.GetY();
   /* normalize distance to goal by looking at the max distance to the goal possible*/
   /* calculate distance to goal from all corners and then compare to get the max */
   float dist1 = Sqrt((maxX - goalX)*(maxX - goalX)+(maxY - goalY)*(maxY - goalY));
   float dist2 = Sqrt((minX - goalX)*(minX - goalX)+(maxY - goalY)*(maxY - goalY));
   float dist3 = Sqrt((maxX - goalX)*(maxX - goalX)+(minY - goalY)*(minY - goalY));
   float dist4 = Sqrt((minX - goalX)*(minX - goalX)+(minY - goalY)*(minY - goalY));
   float maxDist = dist1;
   if(dist2 > maxDist){
     maxDist = dist2;
   }
   if(dist3 > maxDist){
     maxDist = dist3;
   }
   if(dist4 > maxDist){
     maxDist = dist4;
   }
   vecParams.push_back(maxDist);



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

void CCollectiveRLTransport::ZMQSendFailures() {
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
void CCollectiveRLTransport::ZMQSendRewards(){

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
void CCollectiveRLTransport::ZMQSendForceStats(){
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
void CCollectiveRLTransport::ZMQSendObjectStats(){
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
void CCollectiveRLTransport::ZMQSendRobotStats(){
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
void CCollectiveRLTransport::ZMQSendObjectStatsFinal(){
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
void CCollectiveRLTransport::ZMQSendGateStats(){
  if (zmq_send_const(
    m_ptZMQSocket,                        // The socket
    const_cast <float*>(&m_vecGateStats[0]),  // data pointer
    sizeof(float)*m_vecGateStats.size(),      // data size in bytes
    0)                                    //final message
  < 0) {                                  // >= 0 means success
    THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
  }
}



/****************************************/
/****************************************/
void CCollectiveRLTransport::ZMQSendObstacleStats(){
  if (zmq_send_const(
    m_ptZMQSocket,                        // The socket
    const_cast <float*>(&m_vecObstacleStats[0]),  // data pointer
    sizeof(float)*m_vecObstacleStats.size(),      // data size in bytes
    0)                                    //no special flags
  < 0) {                                  // >= 0 means success
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
