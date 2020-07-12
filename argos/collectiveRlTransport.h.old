#ifndef COLLECTIVE_RL_TRANSPORT_H
#define COLLECTIVE_RL_TRANSPORT_H

#include <buzz/argos/buzz_loop_functions.h>
#include <argos3/core/utility/math/rng.h>
#include "server/ModelServerClient.hpp"

class CCollectiveRlTransport : public CBuzzLoopFunctions {

public:

   CCollectiveRlTransport() {}
   virtual ~CCollectiveRlTransport() {}

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
   /** The output file name */
  std::string m_strOutFile;
  /* the goal position */
  CVector2 m_cGoal;
  /* threshold to say that we are close enough to the goal */
  double m_cThreshold;
  UInt32 m_cActionSize;
  UInt32 m_cObsSize;
  UInt32 m_cNumRobots;
  UInt32 m_cNumEpisodes;
  UInt32 m_cEpisodeTime;
  float m_cTimeOutReward;
  float m_cGoalReward;
  /** The output file stream */
  std::ofstream m_cOutFile;

  /** The Random Number Generator */
  CRandom::CRNG* m_pcRNG;




private:
   void PlaceRobots(const CVector2& cCenter,
                    UInt32 newNRobots,
                    UInt32 startID,
                    bool makeRobots);
  void PlaceCylinder();

  bool IsEpisodeFinished();

  void GetObservations(UInt32 stateType, float* resultBuffer);
  /* Socket */
  ModelServerClient *client;
};

#endif
