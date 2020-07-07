#include "collectiveRlTransport.h"
#include "buzz/buzzvm.h"
#include <argos3/plugins/simulator/entities/cylinder_entity.h>
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>
#include <argos3/core/simulator/simulator.h>
#include <unordered_map>

static const std::string FB_CONTROLLER = "fgc";

bool reachedGoal;
UInt32 experimentCounter;
UInt32 episodeNum = 0;
std::vector<CVector3> *randomPositionOffsets = new std::vector<CVector3>(); //Delete me in Destroy

unsigned int maxTicksLeftEpisode;

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
void CCollectiveRlTransport::Init(TConfigurationNode& t_tree) {
   /* Parse XML tree */
   GetNodeAttribute(t_tree, "data", m_strOutFile);
   GetNodeAttribute(t_tree, "robot_num", m_cNumRobots);
   GetNodeAttribute(t_tree, "goal", m_cGoal);
   GetNodeAttribute(t_tree, "threshold", m_cThreshold);
   GetNodeAttribute(t_tree, "obs_size", m_cObsSize);
   GetNodeAttribute(t_tree, "action_size", m_cActionSize);
   GetNodeAttribute(t_tree, "num_episodes", m_cNumEpisodes);
   GetNodeAttribute(t_tree, "episode_time", m_cEpisodeTime);
   GetNodeAttribute(t_tree, "time_out_reward", m_cTimeOutReward);
   GetNodeAttribute(t_tree, "goal_reward", m_cGoalReward);
   reachedGoal = false;
   experimentCounter = 0;
   maxTicksLeftEpisode = m_cEpisodeTime;
   /* Server Client */
   client = new ModelServerClient(55555, 55555, m_cNumRobots, m_cObsSize, m_cActionSize);
   /* Create a new RNG */
   m_pcRNG = CRandom::CreateRNG("argos");
   /* generating random postions for the cylinder*/
   CRange<SInt32> x(-4, -2);
   CRange<SInt32> y(-2, 2);
   for(int i = 0; i <= m_cNumEpisodes; i++){
     CVector3 pos((m_pcRNG -> Uniform(x)),
             (m_pcRNG -> Uniform(y)),
             0);
     randomPositionOffsets->push_back(pos);
   }
   /* place the cylinder */
   PlaceCylinder();
   /* Get all the cylinder objects in the arena */
   CSpace::TMapPerType& m_cCylinders = GetSpace().GetEntitiesByType("cylinder");
   /* Create a new vector for the position of the cylinder*/
   CVector2 cPos;
   /* For each individual Cylinder */
   for(CSpace::TMapPerType::iterator it = m_cCylinders.begin();
       it != m_cCylinders.end();
       ++it) {
      /* Get handle to Cylinder entity and controller */
      CCylinderEntity& cCylinder = *any_cast<CCylinderEntity*>(it->second);
      /* Set the vector equal to the X and Y position of the cylinder */
      cPos.Set(cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
               cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
   }
   /* Place the robots evenly around the cylinder */
   PlaceRobots(cPos, m_cNumRobots, 0, true);
   /* Call buzz Init() (HAS TO BE THE LAST LINE)*/
   CBuzzLoopFunctions::Init(t_tree);
}

/**********************************
This function will place a cylinder randomly within some range in the environment.
This range could be an input later on.
**********************************/
void CCollectiveRlTransport::PlaceCylinder(){
    CCylinderEntity* m_pcCylinder;

    CRange<SInt32> x(-4, -2);
    CRange<SInt32> y(-1.5, 1.5);
    CVector3 pos((m_pcRNG -> Uniform(x)),
            (m_pcRNG -> Uniform(y)),
            0);
    m_pcCylinder = new CCylinderEntity("c1", pos,
                                       CQuaternion(ToRadians(CDegrees(180)), CVector3::Z),
                                       true,
                                       0.5,
                                       0.25,
                                       100);
    AddEntity(*m_pcCylinder);
}

/**********************************
This function will take in the center of the cylinder, the number of robots,
and the starting ID. It will then place all of the robots evenly around the
cylinder and orient them facing the center of the cylinder
**********************************/
void CCollectiveRlTransport::PlaceRobots(const CVector2& cCenter,
                           UInt32 newNRobots,
                           UInt32 startID,
                           bool makeRobots){
  try{
    /*Find the angle that separates the robots around the cylinder*/
    CRadians separatingAngle = ToRadians(CDegrees(360)/newNRobots);
    if(makeRobots){
      CFootBotEntity* pcFB;
      std::ostringstream cFBId;
      /* For Each Robot */
      for (size_t i = 0; i < newNRobots; i++){
        /* Find the angle for this robot*/
        CRadians angle = i*separatingAngle;
        /*Make the id*/
        cFBId.str("");
        cFBId << "fb"<<(i+startID);
        /* Create the robot around the cylinder and add it to ARGoS space */
        pcFB = new CFootBotEntity(
          cFBId.str(),
          FB_CONTROLLER,
          /* Set the position of the current robot*/
          CVector3(cCenter.GetX() + Cos(angle), cCenter.GetY() + Sin(angle), 0),
          /* Set the orientation of the current robot*/
          CQuaternion(ToRadians(CDegrees(180))+angle, CVector3::Z));
        /* Add the robots to the space */
        AddEntity(*pcFB);
        }
      }
      else{
        CSpace::TMapPerType& m_cFootbots = GetSpace().GetEntitiesByType("foot-bot");
        int i = 0;
        for(CSpace::TMapPerType::iterator it = m_cFootbots.begin();
           it != m_cFootbots.end();
           ++it)
           {
             CFootBotEntity& cFootBot = *any_cast<CFootBotEntity*>(it->second);
             CVector3 cCenter = randomPositionOffsets->at(episodeNum);
             CVector3 robotPosition(cCenter.GetX() + Cos(separatingAngle*i), cCenter.GetY() + Sin(separatingAngle*i), 0);
             if(!MoveEntity(
               cFootBot.GetEmbodiedEntity(),                // move the body of the robot
               robotPosition,                               // to this position
               CQuaternion(ToRadians(CDegrees(180))+separatingAngle*i, CVector3::Z),                                     // with this orientation
               false)) {                                    // this is not a check, leave the robot there
                 LOGERR << "Can't move cylinder" << std::endl;
              }
              i++;
        }

      }
  }
  catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("While placing robots ", ex);
  }

}
/****************************************/
/****************************************/

void CCollectiveRlTransport::Reset() {

}

/****************************************/
/****************************************/

void CCollectiveRlTransport::Destroy() {
  delete client;
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


void CCollectiveRlTransport::GetObservations(UInt32 stateType, float* resultBuffer){
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
        reward = m_cGoalReward;
        done = 1;
        break;
      case 2:
        reward = m_cTimeOutReward;
        done = 1;
        break;
    }
    resultBuffer[id*m_cObsSize + 0] = distRobotGoal;
    resultBuffer[id*m_cObsSize + 1] = (float)(ToDegrees(angRobotGoal).GetValue());
    resultBuffer[id*m_cObsSize + 2] = lWheel;
    resultBuffer[id*m_cObsSize + 3] = rWheel;
    resultBuffer[id*m_cObsSize + 4] = distRobotCylinder;
    resultBuffer[id*m_cObsSize + 5] = (float)(ToDegrees(angRobotCylinder).GetValue());
    resultBuffer[id*m_cObsSize + 6] = distCylinderGoal;
    resultBuffer[id*m_cObsSize + 7] = done;
    resultBuffer[id*m_cObsSize + 8] = reward;
    id ++;
  }
}

/****************************************/
/****************************************/
void CCollectiveRlTransport::PreStep() {
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

   // PUSH DISTANCE AND ANGLE THROUGH SOCKET TO PYTHON
   float *observations = new float[m_cNumRobots*m_cObsSize];
   GetObservations(0, observations);
   client->SendAgentObservations(observations);
   delete observations;

   std::map<std::string, float> mapLIncrease;
   std::map<std::string, float> mapRIncrease;

   CSpace::TMapPerType& m_cFootbots = GetSpace().GetEntitiesByType("foot-bot");

   /* GRAB ACTIONS THROUGH SOCKET FROM PYTHON */
   UInt32 id (0);
   float *tempActions = client->GetAgentActions();
   m_cFootbots = GetSpace().GetEntitiesByType("foot-bot");
   for(CSpace::TMapPerType::iterator it = m_cFootbots.begin();
      it != m_cFootbots.end();
      ++it) {
        float *tempAction = tempActions+(id*m_cActionSize);
        CVector3 action;
        if(tempAction[0] == 0){
          action.SetX(-0.1);
        }
        else if(tempAction[0] == 1){
          action.SetX(0.0);
        }
        else if(tempAction[0] == 2){
          action.SetX(0.1);
        }
        if(tempAction[1] == 0){
          action.SetY(-0.1);
        }
        else if(tempAction[1] == 1){
          action.SetY(0.0);
        }
        else if(tempAction[1] == 2){
          action.SetY(0.1);
        }
        action.SetZ(tempAction[2]);
        mapLIncrease[it->first] = action.GetX(); //PLACE ACTUAL INCREASE HERE
        mapRIncrease[it->first] = action.GetY(); //PLACE ACTUAL INCREASE HERE
        id++;
   }
   BuzzForeachVM(PutIncreases(mapLIncrease, mapRIncrease));
}

/****************************************/
/****************************************/

bool CCollectiveRlTransport::IsExperimentFinished() {
   /* This is where we will check if the object breaks
      or if we have reached the goal position. */
  if(experimentCounter < m_cNumEpisodes){
    return false;
  }
  else{
    LOG<<"Ending Experiment"<<std::endl;
    maxTicksLeftEpisode = m_cEpisodeTime;
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
void CCollectiveRlTransport::PostExperiment() {
  LOG<<"Closing the Server"<<std::endl;
  client->CloseServer();
}

void CCollectiveRlTransport::PostStep() {
  CSpace::TMapPerType& m_cCylinders = GetSpace().GetEntitiesByType("cylinder");
  CVector2 cPos;
  for(CSpace::TMapPerType::iterator it = m_cCylinders.begin();
      it != m_cCylinders.end();
      ++it) {
     /* Get handle to Cylinder entity */
     CCylinderEntity& cCylinder = *any_cast<CCylinderEntity*>(it->second);
     /* Set the new vector equal to the current position of the cylinder */
     cPos.Set(cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
              cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
  }

  if((cPos - m_cGoal).Length() < m_cThreshold) {
    /*PUSH REWARD AND DONE THROUGH SOCKET TO PYTHON*/
    reachedGoal = true;
  }
  /* decrement remaining time */
  maxTicksLeftEpisode--;

  /*If we havnt reached our experiment limit then reset*/
  if(IsEpisodeFinished()){
    argos::CSimulator& cSimulator = argos::CSimulator::GetInstance();
      LOG << "Episode " << episodeNum <<" is done" << std::endl;
    float *observations = new float[m_cNumRobots*m_cObsSize]; // Delete Me somewhere
    GetObservations(reachedGoal ? 1 : 2, observations);
    client->SendAgentObservations(observations);
    delete observations;
    experimentCounter ++;
    maxTicksLeftEpisode = m_cEpisodeTime;
    cSimulator.Reset();
    /* Move cylinder and robots according to random position*/
    CSpace::TMapPerType& m_cCylinders = GetSpace().GetEntitiesByType("cylinder");
    /* Create a new vector for the position of the cylinder*/
    CVector2 cPos;
    /* For each individual Cylinder */
    for(CSpace::TMapPerType::iterator it = m_cCylinders.begin();
        it != m_cCylinders.end();
        ++it) {
          CCylinderEntity& cCylinder = *any_cast<CCylinderEntity*>(it->second);

          if(!MoveEntity(
            cCylinder.GetEmbodiedEntity(),                                       // move the body of the robot
            randomPositionOffsets->at(episodeNum),           // to this position
            CQuaternion(ToRadians(CDegrees(180)), CVector3::Z),                                     // with this orientation
            false)) {                                         // this is not a check, leave the robot there
              LOGERR << "Can't move cylinder" << std::endl;
          }
            cPos.Set(cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                     cCylinder.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
    }
    PlaceRobots(cPos, m_cNumRobots, 0, false);
    episodeNum++;
    cSimulator.Execute();

  }
}

bool CCollectiveRlTransport::IsEpisodeFinished(){
  if(reachedGoal){
    LOG<<"Reached Goal"<<std::endl;
    reachedGoal = false;
    return true;
  }
  else if(maxTicksLeftEpisode == 0){
    LOG<<"We have timed out"<<std::endl;
    return true;
  }
  else{
    return false;
  }
}
/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(CCollectiveRlTransport, "collectiveRlTransport");
