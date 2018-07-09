#ifndef  __DUAL_FUNCTIONAL_H__
#define  __DUAL_FUNCTIONAL_H__

#include<deal.II/base/subscriptor.h>
#include<deal.II/base/point.h>
#include<deal.II/lac/vector.h>
#include<deal.II/dofs/dof_handler.h>
#include<deal.II/fe/mapping_q.h>
#include<iostream>

#include "parameters.h"

namespace DualFunctional
{ 
   //note: in the following the primal_solution is actually the interpolation of primal_solution into the
   //dual space. so it already has the correct size
   template <int dim>
   class DualFunctionalBase: public dealii::Subscriptor
   {
    public:
       virtual void assemble_rhs (const dealii::DoFHandler<dim>& dof_handler,
                                  const dealii::Mapping<dim>&    mapping,
                                  const dealii::Vector<double>&  primal_solution,
                                  dealii::Vector<double>&        rhs_dual) const=0;
   };

//following classes serve as only algorithms to assemble rhs_dual
//NOTE: Once you have created a new DualFunctional template class,
//DO Remember to explicitly instantiation it in dual_functional.cc, 
//otherwise it will cause link error: "undefined reference to ..."  
//====================================================================================
   template<int dim>
   class PointValueEvaluation: public DualFunctionalBase<dim>
   {
    public:
       PointValueEvaluation (const dealii::Point<dim>& evaluation_point);

       virtual void assemble_rhs (const dealii::DoFHandler<dim>& dof_handler,
                                  const dealii::Mapping<dim>&    mapping,
                                  const dealii::Vector<double>&  primal_solution,
                                  dealii::Vector<double>&        rhs_dual) const;
       DeclException1 (ExcEvaluationPointNotFound, 
                       dealii::Point<dim>, 
                       << "The evaluation point"<<arg1
                       << "was not found among the vertices of the present grid. ");
    protected:
       const dealii::Point<dim> evaluation_point;
   };

//====================================================================================
   template<int dim>
   class DragEvaluation: public DualFunctionalBase<dim>
   {
    public: 
       DragEvaluation (int naca_boundary_id/*const Parameters::AllParameters<dim>& parameters*/);

       virtual void assemble_rhs (const dealii::DoFHandler<dim>& dof_handler,
                                  const dealii::Mapping<dim>&    mapping,
                                  const dealii::Vector<double>&  primal_solution,
                                  dealii::Vector<double>&        rhs_dual) const;
    protected:
       //we use parameters here in order to get the corresponding boundary_type of a 
       //specific boundary_id, which is specified by user and known to parameters.
       //const Parameters::AllParameters<dim>& temp_param;
       const int naca_boundary_id;
   };

}

#endif
