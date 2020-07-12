#ifndef COLLECTIVE_RL_TRANSPORT_H
#define COLLECTIVE_RL_TRANSPORT_H

#include <buzz/argos/buzz_loop_functions.h>
#include <argos3/core/utility/math/rng.h>
#include <argos3/plugins/simulator/entities/cylinder_entity.h>
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>
//#include <server/ModelServerClient.hpp>

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

   /* Number of possible actions */
   UInt32 m_unNumActions;

   /* Number of observations */
   UInt32 m_unNumObs;

   /* Number of robots */
   UInt32 m_unNumRobots;

   /* Number of episodes */
   UInt32 m_unNumEpisodes;

   /* Time limit of each episode */
   UInt32 m_unEpisodeTime;

   /* Reward received upon timeout */
   Real m_fTimeOutReward;

   /* Reward received upon reaching goal */
   Real m_fGoalReward;
   
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

   /** The vector of actions */
   std::vector<float> m_vecActions;

   /** ZeroMQ context */
   void* m_ptZMQContext;

   /** ZeroMQ communication socket */
   void* m_ptZMQSocket;

private:
   
   void CreateEntities();
   
   void PlaceEntities(UInt32 un_episode);

   bool CylinderAtTarget();
   
   bool IsEpisodeFinished();

   void GetObservations(EEpisodeState e_state);

   void ZMQSendEpisodeState(EEpisodeState e_state);

   void ZMQSendTermination();

   void ZMQSendParams();

   void ZMQSendObservations();

   void ZMQGetActions();

   void ZMQGetAck();

};

#endif
