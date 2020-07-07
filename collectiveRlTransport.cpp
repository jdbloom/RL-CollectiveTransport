#include "collectiveRlTransport.h"

#include <buzz/buzzvm.h>
#include <argos3/core/simulator/simulator.h>

#include <unordered_map>

static const std::string FB_CONTROLLER = "fgc";
static const Real WALL_THICKNESS            = 0.1;  // m
static const Real CYLINDER_RADIUS           = 0.5;  // m
static const Real CYLINDER_HEIGHT           = 0.25; // m
static const Real CYLINDER_MASS             = 100;  // kg
static const Real FOOTBOT_RADIUS            = 0.085036758f; // m
static const Real ROBOT_CYLINDER_DISTANCE   = 0.6; // m
static const Real CYLINDER_PLACEMENT_RADIUS = ROBOT_CYLINDER_DISTANCE + FOOTBOT_RADIUS / 2;

UInt32 episodeNum = 0;

/****************************************/
/****************************************/

class GetRobotData : public CBuzzLoopFunctions::COperation{
public:
  virtual void operator()(const std::string& str_robot_id,
                          buzzvm_t t_vm){

      buzzobj_t tCount = BuzzGet(t_vm, "count");
      // check if data is the correct type: in this case int
      if(!buzzobj_isfloat(tCount)){
        LOGERR << str_robot_id << ": variable 'count' has wrong type" <<std::endl;
        return;
      }
      // Get the value
      float nCount = buzzobj_getfloat(tCount);
      // Add counter to list
      m_mapCounters[str_robot_id] = nCount;
  }

public:
  std::map<std::string, float> m_mapCounters;
};

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
   PlaceCylinder(0);
   PlaceRobots(0);
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
      GetSpace().GetArenaLimits().GetMin().GetX()     + CYLINDER_PLACEMENT_RADIUS + WALL_THICKNESS,
      GetSpace().GetArenaLimits().GetMax().GetX() / 2 - CYLINDER_PLACEMENT_RADIUS
      );
   CRange<Real> cYRange(
      GetSpace().GetArenaLimits().GetMin().GetY() + CYLINDER_PLACEMENT_RADIUS + WALL_THICKNESS,
      GetSpace().GetArenaLimits().GetMax().GetY() - CYLINDER_PLACEMENT_RADIUS - WALL_THICKNESS
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
         CQuaternion cOrient(-j * cSlice + cOffset, CVector3::Z);
         m_vecRobotOrient.back().push_back(cOrient);
      }
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PlaceCylinder(UInt32 un_episode) {
   if(un_episode >= m_unNumEpisodes) {
      THROW_ARGOSEXCEPTION("Episode " << un_episode << " is beyond the maximum of " << m_unNumEpisodes);
   }
   if(!MoveEntity(m_pcCylinder->GetEmbodiedEntity(),
                  m_vecCylinderPos[un_episode],
                  CQuaternion())) {
      THROW_ARGOSEXCEPTION("Cannot place cylinder at " << m_vecCylinderPos[un_episode]);
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PlaceRobots(UInt32 un_episode){
   if(un_episode >= m_unNumEpisodes) {
      THROW_ARGOSEXCEPTION("Episode " << un_episode << " is beyond the maximum of " << m_unNumEpisodes);
   }
   for(size_t i = 0; i < m_unNumRobots; ++i) {
      if(!MoveEntity(m_vecRobots[i]->GetEmbodiedEntity(),
                     m_vecRobotPos[un_episode][i],
                     m_vecRobotOrient[un_episode][i])) {
         THROW_ARGOSEXCEPTION("Cannot place foot-bot " << m_vecRobots[i]->GetId() << " at " << m_vecRobotPos[un_episode][i]);
      }
   }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Reset() {

}

/****************************************/
/****************************************/

void CCollectiveRLTransport::Destroy() {
   //delete m_pcPyTorch;
}

/****************************************/
/****************************************/

struct PutIncreases : public CBuzzLoopFunctions::COperation {
  PutIncreases(std::map<std::string, float>& map_l_increase,
	       std::map<std::string, float>& map_r_increase) :
    m_mapLIncrease(map_l_increase),
    m_mapRIncrease(map_r_increase)
  {}
  /** The action happens here */
  virtual void operator()(const std::string& str_robot_id,
			  buzzvm_t t_vm) {
    BuzzPut(t_vm, "L_increase", static_cast<float>(m_mapLIncrease[str_robot_id]));
    BuzzPut(t_vm, "R_increase", static_cast<float>(m_mapRIncrease[str_robot_id]));
  }
  std::map<std::string, float> m_mapLIncrease;
  std::map<std::string, float> m_mapRIncrease;
};

/****************************************/
/****************************************/

struct GetWheelSpeeds : public CBuzzLoopFunctions::COperation {
   /** The action happens here */
   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {
      m_mapLWheel[str_robot_id] = buzzobj_getfloat(BuzzGet(t_vm, "L_wheel"));
      m_mapRWheel[str_robot_id] = buzzobj_getfloat(BuzzGet(t_vm, "R_wheel"));
   }
   std::unordered_map<std::string, float> m_mapLWheel;
   std::unordered_map<std::string, float> m_mapRWheel;
};


void CCollectiveRLTransport::GetObservations(UInt32 stateType, float* resultBuffer){
  /* Get all the cylinder objects in the arena */
  CSpace::TMapPerType& m_cCylinders = GetSpace().GetEntitiesByType("cylinder");
  /* Create a new vector for the position of the cylinder*/
  CVector2 cylinderPos;
  /* For each individual Cylinder */
  for(CSpace::TMapPerType::iterator it = m_cCylinders.begin();
      it != m_cCylinders.end();
      ++it) {
     /* Get handle to Cylinder entity and controller */
     CCylinderEntity& cCylinder = *any_cast<CCylinderEntity*>(it->second);
     /* Set the vector equal to the X and Y position of the cylinder */
     cylinderPos.Set(cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
              cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
  }
  SInt32 id (0);
  CSpace::TMapPerType& m_cFootbots = GetSpace().GetEntitiesByType("foot-bot");
  for(CSpace::TMapPerType::iterator it = m_cFootbots.begin();
     it != m_cFootbots.end();
     ++it) {
    /* Get the wheel speeds*/
    GetWheelSpeeds cGWS;
    BuzzForeachVM(cGWS);
    float lWheel = cGWS.m_mapLWheel[it->first];
    float rWheel = cGWS.m_mapRWheel[it->first];
    /* Get handle to foot-bot entity and controller */
    CFootBotEntity& cFootBot = *any_cast<CFootBotEntity*>(it->second);
    CVector2 robotPos;
    robotPos.Set(cFootBot.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
             cFootBot.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
    CQuaternion c_quaternion(cFootBot.GetEmbodiedEntity().GetOriginAnchor().Orientation);
    CRadians cZAngle, cYAngle, cXAngle;
    c_quaternion.ToEulerAngles(cZAngle, cYAngle, cXAngle);

    float reward;
    float done = 0;
    // Get local Vector from Robot to Goal
    CVector2 deltaRobotGoal = m_cGoal - robotPos;
    float distRobotGoal = deltaRobotGoal.Length();
    CRadians angRobotGoal = deltaRobotGoal.Angle() - cZAngle;
    // Get local vector from Robot to Cylinder
    CVector2 deltaRobotCylinder = cylinderPos - robotPos;
    float distRobotCylinder = deltaRobotCylinder.Length();
    CRadians angRobotCylinder = deltaRobotCylinder.Angle() - cZAngle;
    // Get Distance from cylinder to goal
    CVector2 deltaCylinderGoal = m_cGoal - cylinderPos;
    float distCylinderGoal = deltaCylinderGoal.Length();

    switch(stateType){
      case 0:
        reward = -1 + (1/(distRobotGoal*10));
        break;
      case 1:
        reward = m_fGoalReward;
        done = 1;
        break;
      case 2:
        reward = m_fTimeOutReward;
        done = 1;
        break;
    }
    resultBuffer[id*m_unObsSize + 0] = distRobotGoal;
    resultBuffer[id*m_unObsSize + 1] = (float)(ToDegrees(angRobotGoal).GetValue());
    resultBuffer[id*m_unObsSize + 2] = lWheel;
    resultBuffer[id*m_unObsSize + 3] = rWheel;
    resultBuffer[id*m_unObsSize + 4] = distRobotCylinder;
    resultBuffer[id*m_unObsSize + 5] = (float)(ToDegrees(angRobotCylinder).GetValue());
    resultBuffer[id*m_unObsSize + 6] = distCylinderGoal;
    resultBuffer[id*m_unObsSize + 7] = done;
    resultBuffer[id*m_unObsSize + 8] = reward;
    id ++;
  }
}

/****************************************/
/****************************************/

void CCollectiveRLTransport::PreStep() {   
   // /************************************
   //  This function will grab the wheel speeds for each robot
   //  for the current time step.
   // ************************************/

   // /*
   // Observation:
   // for each robot:
   //   Vector between robot and goal
   //   Left and Right wheel speeds
   //   Distance from the cylinder to the goal
   //   Vector from robot to cylinder
   // */

   // // PUSH DISTANCE AND ANGLE THROUGH SOCKET TO PYTHON
   // float *observations = new float[m_unNumRobots*m_unObsSize];
   // GetObservations(0, observations);
   // m_pcPyTorch->SendAgentObservations(observations);
   // delete observations;

   // std::map<std::string, float> mapLIncrease;
   // std::map<std::string, float> mapRIncrease;

   // CSpace::TMapPerType& m_cFootbots = GetSpace().GetEntitiesByType("foot-bot");

   // /* GRAB ACTIONS THROUGH SOCKET FROM PYTHON */
   // UInt32 id (0);
   // float *tempActions = m_pcPyTorch->GetAgentActions();
   // m_cFootbots = GetSpace().GetEntitiesByType("foot-bot");
   // for(CSpace::TMapPerType::iterator it = m_cFootbots.begin();
   //    it != m_cFootbots.end();
   //    ++it) {
   //      float *tempAction = tempActions+(id*m_unActionSize);
   //      CVector3 action;
   //      if(tempAction[0] == 0){
   //        action.SetX(-0.1);
   //      }
   //      else if(tempAction[0] == 1){
   //        action.SetX(0.0);
   //      }
   //      else if(tempAction[0] == 2){
   //        action.SetX(0.1);
   //      }
   //      if(tempAction[1] == 0){
   //        action.SetY(-0.1);
   //      }
   //      else if(tempAction[1] == 1){
   //        action.SetY(0.0);
   //      }
   //      else if(tempAction[1] == 2){
   //        action.SetY(0.1);
   //      }
   //      action.SetZ(tempAction[2]);
   //      mapLIncrease[it->first] = action.GetX(); //PLACE ACTUAL INCREASE HERE
   //      mapRIncrease[it->first] = action.GetY(); //PLACE ACTUAL INCREASE HERE
   //      id++;
   // }
   // BuzzForeachVM(PutIncreases(mapLIncrease, mapRIncrease));
}

/****************************************/
/****************************************/

bool CCollectiveRLTransport::IsExperimentFinished() {
   /* This is where we will check if the object breaks
      or if we have reached the goal position. */
  if(m_unEpisodeCounter < m_unNumEpisodes){
    return false;
  }
  else{
    LOG<<"Ending Experiment"<<std::endl;
    m_unEpisodeTicksLeft = m_unEpisodeTime;
    return true;
  }
}
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

void CCollectiveRLTransport::PostStep() {
  // CSpace::TMapPerType& m_cCylinders = GetSpace().GetEntitiesByType("cylinder");
  // CVector2 cPos;
  // for(CSpace::TMapPerType::iterator it = m_cCylinders.begin();
  //     it != m_cCylinders.end();
  //     ++it) {
  //    /* Get handle to Cylinder entity */
  //    CCylinderEntity& cCylinder = *any_cast<CCylinderEntity*>(it->second);
  //    /* Set the new vector equal to the current position of the cylinder */
  //    cPos.Set(cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
  //             cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
  // }

  // if((cPos - m_cGoal).Length() < m_fThreshold) {
  //   /*PUSH REWARD AND DONE THROUGH SOCKET TO PYTHON*/
  //   m_bReachedGoal = true;
  // }
  // /* decrement remaining time */
  // m_unEpisodeTicksLeft--;

  // /*If we havnt reached our experiment limit then reset*/
  // if(IsEpisodeFinished()){
  //   argos::CSimulator& cSimulator = argos::CSimulator::GetInstance();
  //     LOG << "Episode " << episodeNum <<" is done" << std::endl;
  //   float *observations = new float[m_unNumRobots*m_unObsSize]; // Delete Me somewhere
  //   GetObservations(m_bReachedGoal ? 1 : 2, observations);
  //   m_pcPyTorch->SendAgentObservations(observations);
  //   delete observations;
  //   m_unEpisodeCounter ++;
  //   m_unEpisodeTicksLeft = m_unEpisodeTime;
  //   cSimulator.Reset();
  //   /* Move cylinder and robots according to random position*/
  //   CSpace::TMapPerType& m_cCylinders = GetSpace().GetEntitiesByType("cylinder");
  //   /* Create a new vector for the position of the cylinder*/
  //   CVector2 cPos;
  //   /* For each individual Cylinder */
  //   for(CSpace::TMapPerType::iterator it = m_cCylinders.begin();
  //       it != m_cCylinders.end();
  //       ++it) {
  //         CCylinderEntity& cCylinder = *any_cast<CCylinderEntity*>(it->second);

  //         if(!MoveEntity(
  //           cCylinder.GetEmbodiedEntity(),                                       // move the body of the robot
  //           m_vecCylinderPos->at(episodeNum),           // to this position
  //           CQuaternion(ToRadians(CDegrees(180)), CVector3::Z),                                     // with this orientation
  //           false)) {                                         // this is not a check, leave the robot there
  //             LOGERR << "Can't move cylinder" << std::endl;
  //         }
  //           cPos.Set(cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
  //                    cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
  //   }
  //   PlaceRobots(cPos, m_unNumRobots, 0, false);
  //   episodeNum++;
  //   cSimulator.Execute();

  // }
}

bool CCollectiveRLTransport::IsEpisodeFinished(){
   return false;
  // if(m_bReachedGoal){
  //   LOG<<"Reached Goal"<<std::endl;
  //   m_bReachedGoal = false;
  //   return true;
  // }
  // else if(m_unEpisodeTicksLeft == 0){
  //   LOG<<"We have timed out"<<std::endl;
  //   return true;
  // }
  // else{
  //   return false;
  // }
}
/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(CCollectiveRLTransport, "collective_rl_transport");
