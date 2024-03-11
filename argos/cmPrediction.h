#ifndef CM_PREDICTION_H
#define CM_PREDICTION_H

#include <buzz/argos/buzz_loop_functions.h>
#include <argos3/core/utility/math/rng.h>
#include <argos3/plugins/simulator/entities/cylinder_entity.h>
#include <argos3/plugins/simulator/entities/box_entity.h>
#include <argos3/plugins/robots/kheperaiv/simulator/kheperaiv_entity.h> // TODO : Change this to khepera's entity correctly and update argos3 library
#include <argos3/plugins/robots/generic/control_interface/ci_proximity_sensor.h>
#include <argos3/core/control_interface/ci_controller.h>
#include <argos3/plugins/robots/kheperaiv/control_interface/ci_kheperaiv_proximity_sensor.h> // TODO: Change to kheperaiv correctly
#include <argos3/plugins/robots/kheperaiv/control_interface/ci_kheperaiv_gripper_force_sensor.h>
#include <argos3/plugins/robots/kheperaiv/simulator/kheperaiv_gripper_entity.h>
#include <argos3/plugins/simulator/entities/proximity_sensor_equipped_entity.h>
#include <argos3/core/utility/networking/tcp_socket.h>
#ifdef ARGOS_COMPILE_QTOPENGL
  #include <argos3/plugins/simulator/visualizations/qt-opengl/qtopengl_user_functions.h>
#endif
//#include <server/ModelServerClient.hpp>
using namespace argos;

class CCMPrediction : public CBuzzLoopFunctions {

public:

   CCMPrediction();
   virtual ~CCMPrediction() {}

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

   /* threshold to say that we are close enough to finding the CM */
   Real m_fThreshold;

   /* number of episodes to tell when to decrease threshold */
   UInt32 m_unDecThresholdTime;

   /* Number of ticks before there is a direction change */
   UInt32 m_unTicksPerDuration;

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

   /* Reward received on how good the prediction is */
   Real m_fPredictionReward;

   /** The Random Number Generator */
   CRandom::CRNG* m_pcRNG;

   /** Have we found the CM within the threshold? */
   bool m_bFoundCM;

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

   /** The box */
   CBoxEntity* m_pcBox;

   /** The networking socket */
   CTCPSocket* socket;

   /** List of robots */
   std::vector<CKheperaIVEntity*> m_vecRobots;

   /** The vector of observations */
   std::vector<float> m_vecObs;

   /** The vector of failure flags*/
   std::vector<int> m_vecFailures;

   /** The vector of rewards */
   std::vector<float> m_vecRewards;

   /** The axel length of the kheperaiv*/
   Real m_fKheperaIVAxelLength;

   /** The radius of the wheel of the kheperaiv*/
   Real m_fKheperaIVWheelRadius;

   /** Number of stats per robot */
   UInt32 m_unNumStats;

   /** The vector of robot stats */
   std::vector<float> m_vecStats;

   /** The vector of object stats*/
   std::vector<float> m_vecObjStats;

   /** The vector of robot stats */
   std::vector<float> m_vecRobotStats;

   /** The vector of object stats */
   std::vector<float> m_vecObjectStats;

   /** The vector of actions */
   std::vector<float> m_vecActions;

   /** ZeroMQ context */
   void* m_ptZMQContext;

   /** ZeroMQ communication socket */
   void* m_ptZMQSocket;

   /** List of robot failure times: -1 = no failure **/
   std::vector <std::vector<SInt32>> m_vecRobotFailures;

   /** vector to keep track of the current offset (not used other than to print) */
   std::vector<Real> m_vecOffset;

   /** Flag for whether or not to use learning or the base model*/
   UInt32 m_unBaseModel;

   /** Are we simulating the robots or taking them from the real field*/
   bool m_bSimulateRobots;

   /** Do we want the object we are grabbing's mass to be offset? Default to false */
   bool m_bSimulateObjectMO;

   /** What robots are we simulating, please put them in clockwise order around the object or I will die */
   std::string m_strRobotsUsed = "0,1,2,3";

   std::vector<Real> m_xOffsetFromRobot;

   std::vector<Real> m_yOffsetFromRobot;

   CRadians Intended_Dir;

   Real wheel_gain = 0.035;

   Real min_wheel_speed = 2;



private:
   /** Takes care of object simulation as well */
   void SimulateRobots();

   void PlaceRobots(UInt32 un_episode);

   std::vector<SInt32> GenerateRobotFailure();

   bool FoundCM(Real fXCM, Real fYCM);

   bool IsEpisodeFinished();

   Real PredictionDistance(int robot_index);

   void GetObservations(EEpisodeState e_state);

   void CalculateRobotStats();

   void ZMQSendEpisodeState(EEpisodeState e_state);

   void ZMQSendTermination();

   void ZMQSendParams();

   void ZMQSendObservations();

   void ZMQSendFailures();

   void ZMQSendRewards();

   void ZMQSendForceStats();

   void ZMQGetActions();

   void ZMQSendObjectStats();

   void ZMQSendObjectStatsFinal();

   void ZMQSendRobotStats();

   void ZMQGetAck();

};

#endif
