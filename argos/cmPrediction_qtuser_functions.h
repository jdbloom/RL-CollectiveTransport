#ifndef CM_PREDICTION_QTUSER_FUNCTIONS_H
#define CM_PREDICTION_QTUSER_FUNCTIONS_H

#include <buzz/argos/buzz_qt.h>
#include <argos3/plugins/robots/kheperaiv/simulator/kheperaiv_entity.h>
#include <argos3/plugins/simulator/entities/cylinder_entity.h>
#include <argos3/plugins/simulator/entities/box_entity.h>
#include <argos3/plugins/robots/kheperaiv/simulator/dynamics2d_kheperaiv_model.h>
#include <argos3/plugins/robots/kheperaiv/simulator/kheperaiv_measures.h>

using namespace argos;

class CCMPrediction;

class CCMPredictionQTUserFunctions : public CBuzzQT {

public:
  CCMPredictionQTUserFunctions();

  virtual ~CCMPredictionQTUserFunctions(){}

  virtual void DrawInWorld();

  void Draw(CKheperaIVEntity& cKheperaIV);

private:

  CCMPrediction& m_cCRLLF;

};

#endif
