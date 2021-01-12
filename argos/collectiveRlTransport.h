#ifndef COLLECTIVE_RL_TRANSPORT_H
#define COLLECTIVE_RL_TRANSPORT_H

#include <buzz/argos/buzz_loop_functions.h>
#include <argos3/core/utility/math/rng.h>
#include <argos3/plugins/simulator/entities/cylinder_entity.h>
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>
#include <argos3/plugins/robots/generic/control_interface/ci_proximity_sensor.h>
#include <argos3/core/control_interface/ci_controller.h>
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_proximity_sensor.h>
#include <argos3/plugins/simulator/entities/proximity_sensor_equipped_entity.h>
#ifdef ARGOS_COMPILE_QTOPENGL
  #include <argos3/plugins/simulator/visualizations/qt-opengl/qtopengl_user_functions.h>
#endif
//#include <server/ModelServerClient.hpp>
using namespace argos;

class CCollectiveRLTransport : public CBuzzLoopFunctions {

public:

   CCollectiveRLTransport();
   virtual ~CCollectiveRLTransport() {}

   /**
    * Executes user-defined initialization logic.
    * @param t_tree The 'loop_functions' XML configuration tree.
    */
   virtual void Init(TConfigurationNode& t_tree);

   /**
    * Executes user-defined reset logic.
    * This method should restore the state of the simulation at it was right
    * after Init() was called.
    * @see Init()
    */
   virtual void Reset();

   /**
    * Executes user-defined logic right after a control step is executed.
    */
   virtual void PreStep();

   /**
      This function will grab the wheel speed for each robot from the
      learning algorithm.
   */
   virtual void PostStep();

   virtual void PostExperiment();

   /**
    * Returns true if the experiment is finished, false otherwise.
    *
    * This method allows the user to specify experiment-specific ending
    * conditions. If this function returns false and a time limit is set in the
    * .argos file, the experiment will reach the time limit and end there. If no
    * time limit was set, then this function is the only ending condition.
    *
    * @return true if the experiment is finished.
    */
   virtual bool IsExperimentFinished();

   /**
    * Executes user-defined destruction logic.
    * This method should undo whatever is done in Init().
    * @see Init()
    */
   virtual void Destroy();

   /**
   * Returns the position of the goal and the current threshold
   *
   * For use in drawing the threshold on the floor
   */
   inline CVector3 GetGoal() const {
     return CVector3(m_cGoal.GetX(), m_cGoal.GetY(), m_fThreshold);
   }

private:

   enum EEpisodeState {
      EPISODE_RUNNING,
      EPISODE_SUCCESS,
      EPISODE_TIMEOUT
   };

private:

   /** The output file name */
   std::string m_strOutFile;

   /** The output file stream */
   std::ofstream m_cOutFile;

   /* the goal position */
   CVector2 m_cGoal;

   /* threshold to say that we are close enough to the goal */
   Real m_fThreshold;

   /* number of episodes to tell when to decrease threshold */
   UInt32 m_unDecThresholdTime;

   /* amount to decrease threshold by */
   Real m_fDecThreshold;

   /* minimum threshold*/
   Real m_fMinThreshold;

   /* Number of possible actions */
   UInt32 m_unNumActions;

   /* Number of observations */
   UInt32 m_unNumObs;

   /* Number of robots */
   UInt32 m_unNumRobots;

   /* Alphabet size of robot communications */
   UInt32 m_unAlphabetSize;

   /** range for the proximity sensors*/
   Real m_fProximityRange;

   /* Number of episodes */
   UInt32 m_unNumEpisodes;

   /* Time limit of each episode */
   UInt32 m_unEpisodeTime;

   /* Reward received upon timeout */
   Real m_fTimeOutReward;

   /* Reward received upon reaching goal */
   Real m_fGoalReward;

   /*Max Number of robot failures*/
   UInt32 m_unMaxRobotFailures;

   /* Latest time at which a robot can fail */
   UInt32 m_unLatestFailureTime;

   /*Failure Risk Per Robot*/
   Real m_fChanceFailure;

   /** The Random Number Generator */
   CRandom::CRNG* m_pcRNG;

   /** Whether the cylinder reached the goal */
   bool m_bReachedGoal;

   /** Number of episodes */
   UInt32 m_unEpisodeCounter;

   /** Number of time steps left */
   unsigned int m_unEpisodeTicksLeft;

   /** Initial cylinder positions (index = # episode) */
   std::vector<CVector3> m_vecCylinderPos;

   /** Position of the cylinder from the previous time step*/
   CVector3 m_cOldCylinderPos;

   /** Initial robot positions (index = # episode, # robot) */
   std::vector< std::vector<CVector3> > m_vecRobotPos;

   /** Initial robot orientations (index = # episode, # robot) */
   std::vector< std::vector<CQuaternion> > m_vecRobotOrient;

   /** The cylinder */
   CCylinderEntity* m_pcCylinder;

   /** List of robots */
   std::vector<CFootBotEntity*> m_vecRobots;

   /** The vector of observations */
   std::vector<float> m_vecObs;

   /** The vector of failure flags*/
   std::vector<int> m_vecFailures;

   /** The vector of rewards */
   std::vector<float> m_vecRewards;

   /** The axel length of the footbot*/
   Real m_fFootbotAxelLength;

   /** The radius of the wheel of the footbot*/
   Real m_fFootbotWheelRadius;

   /** Number of stats per robot */
   UInt32 m_unNumStats;

   /** The vector of robot stats */
   std::vector<float> m_vecStats;

   /** The vector of actions */
   std::vector<float> m_vecActions;

   /** ZeroMQ context */
   void* m_ptZMQContext;

   /** ZeroMQ communication socket */
   void* m_ptZMQSocket;

   /** List of robot failure times: -1 = no failure **/
   std::vector <std::vector<SInt32>> m_vecRobotFailures;

   /** Number of obstacles to fill in the environment*/
   UInt32 m_unNumObstacles;

   /** Obstacle Positions (index = # episode, # obstacles) */
   std::vector<std::vector<CVector3> > m_vecObstaclePos;

   /** List of obstacles */
   std::vector<CCylinderEntity*> m_vecObstacles;


private:

   void CreateEntities();

   void PlaceEntities(UInt32 un_episode);

   std::vector<SInt32> GenerateRobotFailure();

   bool CylinderAtTarget();

   bool IsEpisodeFinished();

   void GetObservations(EEpisodeState e_state);

   void CalculateRobotStats();

   void ZMQSendEpisodeState(EEpisodeState e_state);

   void ZMQSendTermination();

   void ZMQSendParams();

   void ZMQSendObservations();

   void ZMQSendFailures();

   void ZMQSendRewards();

   void ZMQSendRobotStats();

   void ZMQGetActions();

   void ZMQGetAck();

};

#endif
