#ifndef CM_PREDICTION_QTUSER_FUNCTIONS_H
#define CM_PREDICTION_QTUSER_FUNCTIONS_H

#include <buzz/argos/buzz_qt.h>

using namespace argos;

class CCMPrediction;

class CCMPredictionQTUserFunctions : public CBuzzQT {

public:
  CCMPredictionQTUserFunctions();

  virtual ~CCMPredictionQTUserFunctions(){}

  virtual void DrawInWorld();

private:

  CCMPrediction& m_cCRLLF;

};

#endif
