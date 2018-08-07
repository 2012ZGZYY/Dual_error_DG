#include<deal.II/base/exceptions.h>

#include "claw.h"
#include "dual_functional.h"
#include "dual_solver.h"

using namespace dealii;

// @sect4{ConservationLaw::compute_refinement_indicators}

// This function is real simple: We don't
// pretend that we know here what a good
// refinement indicator would be. Rather, we
// assume that the <code>EulerEquation</code>
// class would know about this, and so we
// simply defer to the respective function
// we've implemented there:
template <int dim>
void
ConservationLaw<dim>::
compute_refinement_indicators (Vector<double> &refinement_indicators) const
{
   //extract mesh_name from parameters.mesh_filename, i.e. remove '.msh' from the filename.
   std::string mesh_name = parameters.mesh_filename;
   int pos = mesh_name.rfind('.');
   if(pos>0)
      mesh_name = mesh_name.erase(pos);
   std::cout<<"mesh_name is "<<mesh_name<<std::endl;
 
   if(parameters.refinement_indicators == Parameters::Refinement::weighted_residual)
    {
     if(mesh_name == "corner" || mesh_name == "corner_coarse")
      {
       const Point<dim> evaluation_point(5, 2.10);    //used in corner compression flow     

       DualFunctional::PointValueEvaluation<dim> dual_functional(evaluation_point);
    
       DualSolver<dim> dual_solver(triangulation,
                                   mapping(),
                                   fe,
                                   current_solution,
                                   parameters,
                                   dual_functional,
                                   time_iter
                                   //mesh_name
                                   //present_time,
                                   //end_time
                                   );

       dual_solver.compute_DWR_indicators(refinement_indicators);
      }      
     else if(mesh_name == "ringleb_structured" || mesh_name == "ringleb" || mesh_name == "ringleb_structured_refined")
      {
       const Point<dim> evaluation_point(0.3, 1.5);      //used in ringleb's flow. this point is defined myself, so there's no reference to compare with.        

       DualFunctional::PointValueEvaluation<dim> dual_functional(evaluation_point);
    
       DualSolver<dim> dual_solver(triangulation,
                                   mapping(),
                                   fe,
                                   current_solution,
                                   parameters,
                                   dual_functional,
                                   time_iter
                                   //mesh_name
                                   //present_time,
                                   //end_time
                                   );

       dual_solver.compute_DWR_indicators(refinement_indicators);
     }
     else if(mesh_name == "naca" || mesh_name == "naca0012" || mesh_name == "naca_uns")
      {
         //use the drag on airfoil face as objective functional, assemble the rhs of the dual_system
         //note: here dual_functional serve as only algorithm to assemble rhs_dual, so it will not take
         //many arguments.

         //firstly we figure out which boundary_id denote the wall_boundary of naca airfoil
         int naca_boundary_id;
         for(unsigned int boundary_id = 0; boundary_id < parameters.max_n_boundaries; ++boundary_id){
            typename EulerEquations<dim>::BoundaryKind boundary_kind = 
                                             parameters.boundary_conditions[boundary_id].kind; 
            if(boundary_kind == EulerEquations<dim>::BoundaryKind::no_penetration_boundary) 
               naca_boundary_id = boundary_id;
         }
         //check the naca_boundary_id is correctly assigned
         std::cout<<"naca_boundary_id = "<< naca_boundary_id << std::endl;
         //if (naca_boundary_id<0 || ) std::cout<<"the id of wall_boundary is not correct! ";

         DualFunctional::DragEvaluation<dim> dual_functional(naca_boundary_id); 
         
         DualSolver<dim> dual_solver(triangulation,
                                     mapping(),
                                     fe,
                                     current_solution,
                                     parameters,
                                     dual_functional,  
                                     time_iter
                                     );
         dual_solver.compute_DWR_indicators(refinement_indicators);
         
      }
     else
      {
       AssertThrow(false, ExcNotImplemented());
      }
    }
   
   else if(parameters.refinement_indicators == Parameters::Refinement::residual)
      {std::cout<<"not finished!";}
   else if(parameters.refinement_indicators == Parameters::Refinement::kelly)
      {std::cout<<"not finished!";}
   else if(parameters.refinement_indicators == Parameters::Refinement::easy){
      if(parameters.time_step_type == "global")
         EulerEquations<dim>::compute_refinement_indicators (dof_handler,
                                                             mapping(),
                                                             predictor,
                                                             refinement_indicators);
      else
         EulerEquations<dim>::compute_refinement_indicators (dof_handler,
                                                             mapping(),
                                                             current_solution,
                                                             refinement_indicators);
   }
   else
   {
      AssertThrow(false, ExcNotImplemented());
   }
}

template <int dim>
void
ConservationLaw<dim>::refine_grid (const Vector<double> &refinement_indicators)
{
   if(parameters.refinement_indicators == Parameters::Refinement::easy){

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
      for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
      {
         cell->clear_coarsen_flag();
         cell->clear_refine_flag();
      
         if ((cell->level() < parameters.shock_levels) &&
             (std::fabs(refinement_indicators(cell_no)) > parameters.shock_val))
            cell->set_refine_flag();
         else
            if ((cell->level() > 0) &&
                (std::fabs(refinement_indicators(cell_no)) < 0.75*parameters.shock_val))
               cell->set_coarsen_flag();
      }
   }
   else if(parameters.refinement_indicators == Parameters::Refinement::weighted_residual){
      const double refine_fraction_cells = .3,
                   coarsen_fraction_cells = .1;
      GridRefinement::refine_and_coarsen_fixed_number(triangulation, 
                                                      refinement_indicators,
                                                      refine_fraction_cells,
                                                      coarsen_fraction_cells,
                                                      parameters.max_n_cells);
   }
   // Then we need to transfer the
   // various solution vectors from
   // the old to the new grid while we
   // do the refinement. The
   // SolutionTransfer class is our
   // friend here; it has a fairly
   // extensive documentation,
   // including examples, so we won't
   // comment much on the following
   // code. The last three lines
   // simply re-set the sizes of some
   // other vectors to the now correct
   // size:
   std::vector<Vector<double> > transfer_in;
   std::vector<Vector<double> > transfer_out;
   
   transfer_in.push_back(old_solution);
   transfer_in.push_back(predictor);
   
   triangulation.prepare_coarsening_and_refinement();
   
   SolutionTransfer<dim> soltrans(dof_handler);
   soltrans.prepare_for_coarsening_and_refinement(transfer_in);
   
   triangulation.execute_coarsening_and_refinement ();
   
   setup_system ();
   
   {
      Vector<double> new_old_solution(1);
      Vector<double> new_predictor(1);
      
      transfer_out.push_back(new_old_solution);
      transfer_out.push_back(new_predictor);
      transfer_out[0].reinit(dof_handler.n_dofs());
      transfer_out[1].reinit(dof_handler.n_dofs());
   }
   
   soltrans.interpolate(transfer_in, transfer_out);

   old_solution = transfer_out[0];
   predictor = transfer_out[1];
   current_solution = old_solution;
}

//---------------------------------------------------------------------------
// Refine cells near the corner of forward step problem
//---------------------------------------------------------------------------
template <int dim>
void
ConservationLaw<dim>::refine_forward_step ()
{
   const double radius = 0.05;
   const Point<dim> corner(0.6, 0.2);
   
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
   {
      cell->clear_coarsen_flag();
      cell->clear_refine_flag();
   
      Tensor<1,dim> dr = cell->center() - corner;
      if(dr.norm() < radius)
         cell->set_refine_flag();
   }
   
   triangulation.prepare_coarsening_and_refinement();
   triangulation.execute_coarsening_and_refinement ();
}

template class ConservationLaw<2>;
