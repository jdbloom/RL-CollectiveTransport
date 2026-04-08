#include "collectiveRlTransport.h"
#include <buzz/buzzvm.h>
#include <zmq.h>
#include <cmath>
#include <stdexcept>

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
static const Real PRISM_HEIGHT           = 0.25; // m
static const Real PRISM_MASS = 100;
static const  std::vector<CVector2> CONVEX_PRISM_POINTS{CVector2(-0.2,-0.2),CVector2(0.2,-0.2),CVector2(0.2,0.2),CVector2(-0.2,0.4)};

std::vector<Real> PRISM_MASSES{75.0,25.0};
static const std::vector<Real> TEST_PRISM_MASSES{75.0,25.0,15.0};
std::vector<std::vector<CVector2>> COMPOSITE_PRISM_POINTS{{CVector2(-0.2,-0.2),CVector2(0.2,-0.2),CVector2(0.2,0.2),CVector2(-0.2,0.2)}, {CVector2(-0.2,0.2),CVector2(-0.4,0.2),CVector2(-0.5,-0.4),CVector2(-0.2,-0.2)}};
static const  std::vector<std::vector<CVector2>> TEST_PRISM_POINTS{{CVector2(-0.1,-0.2),CVector2(0.1,-0.2),CVector2(0.1,0.2),CVector2(-0.1,0.2)},{CVector2(-0.1,0.2),CVector2(-0.3,0.2),CVector2(-0.3,-0.1),CVector2(-0.1,-0.2)},{CVector2(0.1,-0.2),CVector2(0.3,-0.3),CVector2(0.3,0.0),CVector2(0.1,0.2)}};
static const Real CYLINDER_PLACEMENT_RADIUS = WALL_THICKNESS + ROBOT_CYLINDER_DISTANCE + FOOTBOT_RADIUS;

static const std::string OBS_DESCRIPTIONS[] = {
   "robot2goal_dist",
   "robot2goal_angle",
   "lwheel",
   "rwheel",
   "robot2object_dist",
   "robot2object_angle",
   "object2goal_dist",
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
      GetNodeAttribute(t_tree, "use_prisms", m_unUsePrisms);
      GetNodeAttribute(t_tree, "test_prism", m_unUseTestPrism);
      GetNodeAttribute(t_tree, "random_objs", m_unRandomizeObjects);
      GetNodeAttribute(t_tree, "use_base_model", m_unBaseModel);

      /* Footbot dynamic equation parameters*/
      m_fFootbotAxelLength = 0.14; // m
      m_fFootbotWheelRadius = 0.029112741; // m

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
      if(m_unUsePrisms){
         ZMQSendNumPrisms();
         ZMQSendPrismPoints();
      }
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
      m_vecObjStats.resize(9, 0.0);
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
   /** Choose object*/
   m_unObjectChoice = 0;
   if(m_unUsePrisms == 1) {
       m_unObjectChoice = 2;
   }
   if(m_unRandomizeObjects == 1) {
      CRange<UInt32> ObjectRange(0,3);
      m_unObjectChoice = m_pcRNG->Uniform(ObjectRange);
   }
    
   /* Create the object */
   Real max_length = 0;
   if(m_unObjectChoice == 0){
      m_pcCylinder = new CCylinderEntity(
         "c1",
         CVector3(),
         CQuaternion(),
         true,
         CYLINDER_RADIUS,
         CYLINDER_HEIGHT,
         CYLINDER_MASS);
      AddEntity(*m_pcCylinder);
      max_length = CYLINDER_RADIUS;
      m_cObjCOMOffsetPos = CVector2::ZERO;
   }
   else if(m_unObjectChoice == 1) {
      m_pcConvexPrism = new CConvexPrismEntity(
         "Prism_1",
         CVector3(),
         CQuaternion(),
         true,
         CONVEX_PRISM_POINTS,
         PRISM_HEIGHT,
         PRISM_MASS);
      AddEntity(*m_pcConvexPrism);

      CVector2 origin = CVector2(m_pcConvexPrism->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),m_pcConvexPrism->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
      for(size_t i = 0; i < CONVEX_PRISM_POINTS.size(); i++) {
         Real point_distance = Distance(origin, CONVEX_PRISM_POINTS[i]);
         if(point_distance > max_length) {
            max_length = point_distance;
         }
      }
      m_cObjCOMOffsetPos = GetCoM(CONVEX_PRISM_POINTS);

   }
   else if(m_unObjectChoice == 2) {
      // uses the alternate composite prism object for testing
      if(m_unUseTestPrism == 1) {
         PRISM_MASSES = TEST_PRISM_MASSES;
         COMPOSITE_PRISM_POINTS = TEST_PRISM_POINTS;
      }
      m_pcComposite = new CCompositeEntity(
         "Prism_1",
         CVector3(),
         CQuaternion(),
         true,
         PRISM_MASSES,
         COMPOSITE_PRISM_POINTS,
         PRISM_HEIGHT);
      AddEntity(*m_pcComposite);

      CVector2 origin = CVector2(m_pcComposite->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),m_pcComposite->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

      for(size_t i = 0; i < COMPOSITE_PRISM_POINTS.size(); i++) {
         for(size_t j = 0; j < COMPOSITE_PRISM_POINTS[0].size(); j++) {
            Real point_distance = Distance(origin, COMPOSITE_PRISM_POINTS[i][j]);
            LOG<<"point: " << point_distance<<std::endl;
            if(point_distance > max_length) {
               max_length = point_distance;
            }
         }
      }
      m_cObjCOMOffsetPos = GetCoMComposite(PRISM_MASSES,COMPOSITE_PRISM_POINTS);

   }
   
   /* Create robots */
   CRadians cSlice = CRadians::TWO_PI / m_unNumRobots;
   std::ostringstream cFBId;
   CFootBotEntity* pcFB;
   CVector3 cPos;

   
   Real ROBOT_PRISM_DISTANCE   = 2*max_length;
   
   // Adding an offset just makes the robots start slightly farther away
   Real PRISM_PLACEMENT_RADIUS = WALL_THICKNESS + ROBOT_PRISM_DISTANCE + FOOTBOT_RADIUS;


   for(size_t i = 0; i < m_unNumRobots; ++i) {
      cFBId.str("");
      cFBId << "fb" << i;
      cPos.FromSphericalCoords(ROBOT_PRISM_DISTANCE,
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
   // LOG << "Initializing Cylinder X"<<std::endl;
   // LOG << "Min Space Lim " << GetSpace().GetArenaLimits().GetMin().GetX() + PRISM_PLACEMENT_RADIUS <<std::endl;
   // LOG << "Max Space Lim " << GetSpace().GetArenaLimits().GetMin().GetX()/2 -PRISM_PLACEMENT_RADIUS <<std::endl;
   // LOG << "Max Space X / 2 " << GetSpace().GetArenaLimits().GetMin().GetX()/2 <<std::endl;
   // LOG << "Max Space X " << GetSpace().GetArenaLimits().GetMin().GetX() <<std::endl;
   // LOG << "Prism Radius " << PRISM_PLACEMENT_RADIUS <<std::endl;
   
   CRange<Real> cXCylinderRange(
      GetSpace().GetArenaLimits().GetMin().GetX() + PRISM_PLACEMENT_RADIUS,
      GetSpace().GetArenaLimits().GetMin().GetX()/2
      );
   // LOG << "Initializing Obstacle X"<<std::endl;
   CRange<Real> cXObstacleRange(
      GetSpace().GetArenaLimits().GetMin().GetX()/2,
      0
   );
   // LOG << "Initializing Cylinder Y"<<std::endl;
   CRange<Real> cYRange(
      -2 + PRISM_PLACEMENT_RADIUS,
      2 - PRISM_PLACEMENT_RADIUS
      );
   LOG << "Initializing Cylinder Pos"<<std::endl;
   for(size_t i = 0; i < m_unNumEpisodes; ++i){
      auto x_pos = m_pcRNG->Uniform(cXCylinderRange);
      auto y_pos = m_pcRNG->Uniform(cYRange);
      if (x_pos < GetSpace().GetArenaLimits().GetMin().GetX() || x_pos > GetSpace().GetArenaLimits().GetMax().GetX()){
         LOG << "X value is outside of acceptable range: " << x_pos <<std::endl;
         throw std::out_of_range("X Value is outside the acceptable range");
      }
      if (y_pos < GetSpace().GetArenaLimits().GetMin().GetY() || y_pos > GetSpace().GetArenaLimits().GetMax().GetY()){
         LOG << "Y value is outside of acceptable range: " << y_pos <<std::endl;
         throw std::out_of_range("Y Value is outside the acceptable range");
      }
      cPos.Set(x_pos,
               y_pos,
               0.0);
      EnforceBoundaries(cPos, i, "Cylinder");
      m_vecObjectPos.push_back(cPos);
      //Generate Failure Times for all episodes
      m_vecRobotFailures.push_back(GenerateRobotFailure());
   }
   LOG << "Initializing Robot Pos"<<std::endl;
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
         cPos += m_vecObjectPos[i];
         auto x_pos = cPos.GetX();
         auto y_pos = cPos.GetY();
         if (x_pos < GetSpace().GetArenaLimits().GetMin().GetX() || x_pos > GetSpace().GetArenaLimits().GetMax().GetX()){
            LOG << "X value is outside of acceptable range: " << x_pos <<std::endl;
            throw std::out_of_range("X Value is outside the acceptable range");
         }
         if (y_pos < GetSpace().GetArenaLimits().GetMin().GetY() || y_pos > GetSpace().GetArenaLimits().GetMax().GetY()){
            LOG << "Y value is outside of acceptable range: " << y_pos <<std::endl;
            throw std::out_of_range("Y Value is outside the acceptable range");
         }
         /* Add position */
         EnforceBoundaries(cPos, i, "Robot");
         m_vecRobotPos.back().push_back(cPos);
         /* Calculate orientation to cylinder center */
         CQuaternion cOrient(j * cSlice + cOffset + CRadians::PI, CVector3::Z);
         m_vecRobotOrient.back().push_back(cOrient);
      }
   }
   LOG << "Initializing Obstacle Pos"<<std::endl;
   /** Generate Random Positions for the obstacles */
   CRange<Real> cYObstacleRange(
      GetSpace().GetArenaLimits().GetMin().GetY() + OBSTACLE_RADIUS,
      GetSpace().GetArenaLimits().GetMax().GetY() - OBSTACLE_RADIUS
      );
   for(size_t i = 0; i < m_unNumEpisodes; ++i){
      m_vecObstaclePos.push_back(std::vector<CVector3>());
      for(size_t j = 0; j < m_unNumObstacles; j++){
         auto x_pos = m_pcRNG->Uniform(cXObstacleRange);
         auto y_pos = m_pcRNG->Uniform(cYObstacleRange); 
         if (x_pos < GetSpace().GetArenaLimits().GetMin().GetX() || x_pos > GetSpace().GetArenaLimits().GetMax().GetX()){
            LOG << "X value is outside of acceptable range: " << x_pos <<std::endl;
            throw std::out_of_range("X Value is outside the acceptable range");
         }
         if (y_pos < GetSpace().GetArenaLimits().GetMin().GetY() || y_pos > GetSpace().GetArenaLimits().GetMax().GetY()){
            LOG << "Y value is outside of acceptable range: " << y_pos <<std::endl;
            throw std::out_of_range("Y Value is outside the acceptable range");
         }
         cPos.Set(x_pos,
                y_pos,
                0.0);
         EnforceBoundaries(cPos, i, "Obstacle");
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
         EnforceBoundaries(cPos, i, "Gate Lower");
         m_vecGateWallPos.back().push_back(cPos);
         cPos.Set(XPos,
                  GetSpace().GetArenaLimits().GetMax().GetY() - (abs(GetSpace().GetArenaLimits().GetMax().GetY() - (YPos + offset)))/2,
                  0.0);
         EnforceBoundaries(cPos, i, "Gate Upper");
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

void CCollectiveRLTransport::EnforceBoundaries(CVector3& pos, size_t episode, std::string state){
   const CRange<CVector3>& cArenaLimits = GetSpace().GetArenaLimits();

   Real buffer = 0.1;

   if (pos.GetX() < cArenaLimits.GetMin().GetX() + buffer){
      LOG<<"Episode " << episode << " Errored Placing " << state<< std::endl;
      LOG << "Position X is OUT OF BOUNDS: Min" << std::endl;
      pos.SetX(cArenaLimits.GetMin().GetX() + buffer);
   }
   else if (pos.GetX() > cArenaLimits.GetMax().GetX() - buffer){
      LOG<<"Episode " << episode << " Errored Placing " << state<< std::endl;
      LOG << "Position X is OUT OF BOUNDS: Max" << std::endl;
      pos.SetX(cArenaLimits.GetMax().GetX() - buffer);
   }
   if (pos.GetY() < cArenaLimits.GetMin().GetY() + buffer){
      LOG<<"Episode " << episode << " Errored Placing " << state<< std::endl;
      LOG << "Position Y is OUT OF BOUNDS: Min" << std::endl;
      pos.SetY(cArenaLimits.GetMin().GetY() + buffer);
   }
   else if (pos.GetY() > cArenaLimits.GetMax().GetY() - buffer){
      LOG<<"Episode " << episode << " Errored Placing " << state<< std::endl;
      LOG << "Position Y is OUT OF BOUNDS: Max" << std::endl;
      pos.SetY(cArenaLimits.GetMax().GetY() - buffer);
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PlaceEntities(UInt32 un_episode) {
   /* Make sure the episode is valid */
   if(un_episode >= m_unNumEpisodes) {
      THROW_ARGOSEXCEPTION("Episode " << un_episode << " is beyond the maximum of " << m_unNumEpisodes);
   }
   /* Get the old position of the object*/
   m_cOldObjectPos = m_vecObjectPos[un_episode];
   /* The placements we chose are collision-free by construction, no need to
    * check for collisions */
   if(m_unObjectChoice == 0) {
      LOG << "Placing Cylinder"<<std::endl;
      MoveEntity(m_pcCylinder->GetEmbodiedEntity(), // body
              m_vecObjectPos[un_episode],      // position
              CQuaternion(),                     // orientation
              false,                             // not a check
              true);                             // ignore collisions
   }
   else if(m_unObjectChoice == 1) {
      LOG << "Placing Convex Prism"<<std::endl;
      MoveEntity(m_pcConvexPrism->GetEmbodiedEntity(), // body
              m_vecObjectPos[un_episode],      // position
              CQuaternion(),                     // orientation
              false,                             // not a check
              true);                             // ignore collisions
   }
   else if(m_unObjectChoice == 2) {
      LOG << "Placing Compisite"<<std::endl;
      MoveEntity(m_pcComposite->GetEmbodiedEntity(), // body
              m_vecObjectPos[un_episode],      // position
              CQuaternion(),                     // orientation
              false,                             // not a check
              true);                             // ignore collisions
   }
   
   for(size_t i = 0; i < m_vecRobots.size(); ++i) {
      LOG << "Placing Robot " << i <<std::endl;
      MoveEntity(m_vecRobots[i]->GetEmbodiedEntity(), // body
                 m_vecRobotPos[un_episode][i],        // position
                 m_vecRobotOrient[un_episode][i],     // orientation
                 false,                               // not a check
                 true);                               // ignore collisions
   }
   for(size_t i = 0; i < m_vecObstacles.size(); ++i){
      LOG << "Placing Obstacle" << i <<std::endl;
     MoveEntity(m_vecObstacles[i]->GetEmbodiedEntity(), // body
                m_vecObstaclePos[un_episode][i],        // position
                CQuaternion(),                          // orientation
                false,                                  // not a check
                true);                                  // ignore collisions
   }
   if(m_unUseGate == 1){
      /** Resize the walls */
      m_vecGateWalls[0]->Resize(m_vecGateWallSize[un_episode][0]);
      m_vecGateWalls[1]->Resize(m_vecGateWallSize[un_episode][1]);
      /** Move the Walls */
      LOG << "Placing Gate"<<std::endl;
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
    }


}
/****************************************/
/****************************************/

CVector2 CCollectiveRLTransport::GetCoM(std::vector<CVector2> vec_vertices) {
   CVector2 sum = CVector2::ZERO;
   for(size_t i = 0; i < vec_vertices.size(); ++i) {
      sum += vec_vertices[i];
   }
   CVector2 com_offset = sum / vec_vertices.size();
   return com_offset;
}

/****************************************/
/****************************************/

CVector2 CCollectiveRLTransport::GetCoMComposite(std::vector<Real> vec_masses, std::vector<std::vector<CVector2>> vec_vertices) {
   std::vector<CVector2> prism_coms;
   for(size_t i = 0; i < vec_vertices.size(); ++i) {
      CVector2 prism_com = GetCoM(vec_vertices[i]);
      prism_coms.push_back(prism_com);
   }
   Real mass_sum = vec_masses[0];
   CVector2 overall_com = prism_coms[0];
   for(size_t i = 1; i < prism_coms.size(); ++i) {
      Real new_mass_sum = mass_sum + vec_masses[i];
      overall_com = (overall_com*mass_sum + prism_coms[i]*vec_masses[i])/new_mass_sum;
      mass_sum = new_mass_sum; 
   }

   return overall_com;
}

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
   LOG <<"Done Placing Entitiies"<<std::endl;
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
   CVector3 cObjectPos = CVector3::ZERO;
   CQuaternion cCylinderOrient;
   if(m_unObjectChoice == 0) {
      cObjectPos = m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position;
      cCylinderOrient = m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Orientation;       
   }
   else if(m_unObjectChoice == 1) {
      cObjectPos = m_pcConvexPrism->GetEmbodiedEntity().GetOriginAnchor().Position;
      cCylinderOrient = m_pcConvexPrism->GetEmbodiedEntity().GetOriginAnchor().Orientation;  
   }
   else if(m_unObjectChoice == 2) {
      cObjectPos = m_pcComposite->GetEmbodiedEntity().GetOriginAnchor().Position;
      cCylinderOrient = m_pcComposite->GetEmbodiedEntity().GetOriginAnchor().Orientation;  
   }
   
   CRadians cObjZ, cObjY, cObjX;
   cCylinderOrient.ToEulerAngles(cObjZ, cObjY, cObjX);
   /** Store object position and orientation to send to python*/
   CVector2 COMPos = CVector2(m_cObjCOMOffsetPos.GetX() + cObjectPos.GetX(), m_cObjCOMOffsetPos.GetY() + cObjectPos.GetY()).Rotate(cObjZ);
   m_vecObjStats[0] = cObjectPos.GetX();
   m_vecObjStats[1] = cObjectPos.GetY();
   m_vecObjStats[2] = cObjectPos.GetZ();
   m_vecObjStats[3] = ToDegrees(cObjX).GetValue();
   m_vecObjStats[4] = ToDegrees(cObjY).GetValue();
   m_vecObjStats[5] = ToDegrees(cObjZ).GetValue();
   m_vecObjStats[7] = COMPos.GetX();
   m_vecObjStats[8] = COMPos.GetY();
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
      /* Get vector from robot to object (robot-local) */
      CVector2 cVecRobot2Object(
         cObjectPos.GetX() - cRobotPos.GetX(),
         cObjectPos.GetY() - cRobotPos.GetY());
      cVecRobot2Object.Rotate(-cRobotZ);
      /* Get vector from object to goal (robot-local) */
      CVector2 cVecObject2Goal
         (m_cGoal.GetX() - cObjectPos.GetX(), m_cGoal.GetY() - cObjectPos.GetY());
      /* Get the cosine similarity of the object */
      CVector2 cMotion(cObjectPos.GetX() - m_cOldObjectPos.GetX(), cObjectPos.GetY() - m_cOldObjectPos.GetY());
      Real fDirection = 0;
      if(cMotion.Length()>1e-4){
        fDirection= cVecObject2Goal.DotProduct(cMotion) /
                          (cVecObject2Goal.Length()*cMotion.Length());

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
      m_vecObs[i * m_unNumObs + 4] = cVecRobot2Object.Length();
      m_vecObs[i * m_unNumObs + 5] = ToDegrees(cVecRobot2Object.Angle()).GetValue();
      m_vecObs[i * m_unNumObs + 6] = cVecObject2Goal.Length();
      /* Adding object angle to goal for MME*/
      m_vecObjStats[6] = ToDegrees(cVecObject2Goal.Angle()).GetValue();
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
   
   if(m_unObjectChoice == 0) {
      m_cOldObjectPos = m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position;   
   }
   else if(m_unObjectChoice == 1) {
      m_cOldObjectPos = m_pcConvexPrism->GetEmbodiedEntity().GetOriginAnchor().Position;
   }
   else if(m_unObjectChoice == 2) {
      m_cOldObjectPos = m_pcComposite->GetEmbodiedEntity().GetOriginAnchor().Position; 
   }

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
   /* Check if the object reached the goal */
   m_bReachedGoal = ObjectAtTarget();
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

bool CCollectiveRLTransport::ObjectAtTarget() {
   CVector3 cObjectPos = CVector3::ZERO;
   if(m_unObjectChoice == 0) {
      cObjectPos = m_pcCylinder->GetEmbodiedEntity().GetOriginAnchor().Position;   
   }
   else if(m_unObjectChoice == 1) {
      cObjectPos = m_pcConvexPrism->GetEmbodiedEntity().GetOriginAnchor().Position;
   }
   else if(m_unObjectChoice == 2) {
      cObjectPos = m_pcComposite->GetEmbodiedEntity().GetOriginAnchor().Position; 
   }
   CVector2 cObject2Goal(
      m_cGoal.GetX() - cObjectPos.GetX(),
      m_cGoal.GetY() - cObjectPos.GetY());
   return cObject2Goal.Length() < m_fThreshold;
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
   vecParams.push_back(COMPOSITE_PRISM_POINTS.size());

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

void CCollectiveRLTransport::ZMQSendNumPrisms(){
   std::vector<int> m_vecNumPrismPoints;
   for (size_t i = 0; i < COMPOSITE_PRISM_POINTS.size(); ++i){
      m_vecNumPrismPoints.push_back(COMPOSITE_PRISM_POINTS[i].size());
   }
   if(zmq_send_const(
         m_ptZMQSocket,                    // the socket
         const_cast<int*>(&m_vecNumPrismPoints[0]), // data pointer
         sizeof(int) * m_vecNumPrismPoints.size(),  // data size in bytes
         0)                      // another message will follow (rewards)
      < 0) {                               // >= 0 means success
      THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
   }
   ZMQGetAck();
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::ZMQSendPrismPoints(){
   std::vector<float> m_vecPrismPoints;
   for (size_t i = 0; i < COMPOSITE_PRISM_POINTS.size(); ++i){
      for (size_t j = 0; j < COMPOSITE_PRISM_POINTS[i].size(); ++j){
         m_vecPrismPoints.push_back(COMPOSITE_PRISM_POINTS[i][j].GetX());
         m_vecPrismPoints.push_back(COMPOSITE_PRISM_POINTS[i][j].GetY());
      }
   }
   if(zmq_send_const(
         m_ptZMQSocket,                    // the socket
         const_cast<float*>(&m_vecPrismPoints[0]), // data pointer
         sizeof(float) * m_vecPrismPoints.size(),  // data size in bytes
         0)                      // another message will follow (rewards)
      < 0) {                               // >= 0 means success
      THROW_ARGOSEXCEPTION("Cannot send data to PyTorch: " << zmq_strerror(errno));
   }
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
