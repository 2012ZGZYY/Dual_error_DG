#ifndef  __DUAL_SOLVER_H__
#define __DUAL_SOLVER_H__

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h> 
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/precondition_block.h>
/*
#include <deal.II/lac/petsc_solver.h>          //to use PETScWrappers::SolverGMRES
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
*/

#include <deal.II/lac/trilinos_sparse_matrix.h>    //to use TrilinosWrappers

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <vector>

#include "parameters.h"
#include "dual_functional.h" 

template<int dim>
class DualSolver
{
 public:
    DualSolver(const dealii::Triangulation<dim>&               triangulation,
               const dealii::Mapping<dim>&                     mapping,
               const dealii::FESystem<dim>&                    fe,
               const dealii::Vector<double>&                   primal_solution,
               const Parameters::AllParameters<dim>&           parameters,
               const DualFunctional::DualFunctionalBase<dim>&  dual_functional,
               const unsigned int&                             timestep_no
               );
    ~DualSolver();
    //const dealii::Vector<double>& solve_dual_problem();
    void compute_DWR_indicators(dealii::Vector<double>& refinement_indicators, double& estimated_error_obj);
    //double get_estimated_error() const;
 private:
    void setup_system();
    void assemble_matrix();
    
    //following 3 integrate_functions are used in assemble_matrix()
    typedef dealii::MeshWorker::DoFInfo<dim> DoFInfo;
    typedef dealii::MeshWorker::IntegrationInfo<dim> CellInfo;
    void integrate_cell_term(DoFInfo& dinfo, CellInfo& info);
    void integrate_boundary_term(DoFInfo& dinfo, CellInfo& info);
    void integrate_face_term(DoFInfo& dinfo1, DoFInfo& dinfo2, CellInfo& info1, CellInfo& info2);
    
    void integrate_cell_term();
    void assemble_rhs();
    std::pair<unsigned int, double> solve();
    void output_results();      //output the dual_solution, implemented in dual_solver.cc
    

    //following 4 members are used to obtain informations of the primal_problem
    const dealii::Mapping<dim>&     mapping;                   //initialized in constructor
    const dealii::FESystem<dim>&    fe;                        //const reference, initialized in constructor
    const Parameters::AllParameters<dim>&  parameters;         //const reference, initialized in constructor
    dealii::DoFHandler<dim>         dof_handler;               //initialized in constructor
    const dealii::Vector<double>    primal_solution_origin;    //used to store primal_solution in primal_space
    //const unsigned int              fe_degree;

    //following 4 members are used in dual_problem
    const unsigned int              fe_degree_dual;            //initialized in constructor
    const dealii::FESystem<dim>     fe_dual;                   //initialized in constructor, used in setup_system()
    dealii::DoFHandler<dim>         dof_handler_dual;          //initialized in constructor, used in setup_system()   
    dealii::Vector<double>          primal_solution;           //used to store primal_solution in dual_space
    

    const dealii::QGauss<dim>       quadrature_formula;         //used to assemble dual_system and compute
    const dealii::QGauss<dim-1>     face_quadrature_formula;    //the error_indicators 
 
    //following members all used 
    dealii::SparsityPattern             sparsity_pattern_origin;
    dealii::SparsityPattern             sparsity_pattern_dual;     //used in setup_system()
    dealii::SparsityPattern             sparsity_pattern_identity;  //used to generate identity sparse matrix 

    //use dealii's SparseMatrix
    dealii::SparseMatrix<double>              system_matrix_origin;      //used to compute its transpose
    dealii::SparseMatrix<double>              system_matrix_dual;
    dealii::SparseMatrix<double>              system_matrix_identity;
    
    dealii::Vector<double>          system_rhs_dual;           //used in setup_system()
    dealii::Vector<double>          solution_dual;             //used in setup_system()
    dealii::Vector<double>          dual_weights;
    
    
    //const ConservationLaw<dim> cons;

    //used in compute_DWR_indicators() to call "triangulation->n_active_cells()"
    const dealii::SmartPointer<const dealii::Triangulation<dim>>  triangulation; 
    const dealii::SmartPointer<const DualFunctional::DualFunctionalBase<dim>> dual_functional; //used in assemble_rhs()
    
    //double            estimated_error;    //used to store the estimated error in the objective functional
    unsigned int       timestep_no;        //initialized in constructor, used in output_results()

  
    struct Integrator
    {
       Integrator(const dealii::DoFHandler<dim>& dof_handler);
       dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
       dealii::MeshWorker::DoFInfo<dim>            dof_info;
       dealii::MeshWorker::Assembler::MatrixSimple<dealii::SparseMatrix<double>> assembler; 
    }; 

    // Call the appropriate numerical flux function
   template <typename InputVector>
   inline
   void numerical_normal_flux 
   (
      const dealii::Tensor<1,dim>      &normal,
      const InputVector                &Wplus,
      const InputVector                &Wminus,
      //const dealii::Vector<double>     &Aplus,
      //const dealii::Vector<double>     &Aminus,
      std::array<typename InputVector::value_type, EulerEquations<dim>::n_components> &normal_flux
   ) const
   {
      switch(parameters.flux_type)
      {/*
         case Parameters::Flux::lxf:
            EulerEquations<dim>::lxf_flux (normal,
                                           Wplus,
                                           Wminus,
                                           Aplus,
                                           Aminus,
                                           normal_flux);
            break;
        */
         case Parameters::Flux::sw:
            EulerEquations<dim>::steger_warming_flux (normal,
                                                      Wplus,
                                                      Wminus,
                                                      normal_flux);
            break;

         case Parameters::Flux::kfvs:
            EulerEquations<dim>::kfvs_flux (normal,
                                            Wplus,
                                            Wminus,
                                            normal_flux);
            break;
            
         case Parameters::Flux::roe:
            EulerEquations<dim>::roe_flux (normal,
                                           Wplus,
                                           Wminus,
                                           normal_flux);
            break;
            
         case Parameters::Flux::hllc:
            EulerEquations<dim>::hllc_flux (normal,
                                            Wplus,
                                            Wminus,
                                            normal_flux);
            break;

	      default:
            Assert (false, dealii::ExcNotImplemented());
      }
   }


   // Given a cell iterator, return the cell number
   template <typename ITERATOR>
   inline
   unsigned int cell_number (const ITERATOR &cell) const
   {
      return cell->user_index();
   }
   
};

#endif
