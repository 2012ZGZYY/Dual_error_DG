#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

//#include <deal.II/lac/petsc_precondition.h>
//#include <deal.II/lac/petsc_solver.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/precondition.h>   //I add this

#include <deal.II/fe/fe_tools.h>

#include "dual_solver.h"
#include <Sacado.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "equation.h"

using namespace dealii;

template<int dim>
DualSolver<dim>::
DualSolver(const Triangulation<dim>&                       triangulation,
           const dealii::Mapping<dim>&                    mapping,
           const FESystem<dim>&                            fe,
           const Vector<double>&                           primal_solution,
           const Parameters::AllParameters<dim>&           parameters,
           //const ConservationLaw<dim>&                     cons,
           const DualFunctional::DualFunctionalBase<dim>&  dual_functional,
           const unsigned int&                             timestep_no
           //const std::string&                              base_mesh
           //const double&                                   present_time,
           //const double&                                   end_time
           )
           :
           mapping(mapping),
           fe(fe),
           parameters(parameters), 
           dof_handler(triangulation),
           primal_solution_origin(primal_solution),
           fe_degree_dual(fe.degree + 0),
           fe_dual(FE_DGQArbitraryNodes<dim>(QGauss<1>(fe_degree_dual+1)),
                                             EulerEquations<dim>::n_components),  
           dof_handler_dual (triangulation),
           quadrature_formula (fe_degree_dual+1),
           face_quadrature_formula(fe_degree_dual+1),
           //cons (cons),
           triangulation(&triangulation),            //triangulation is a pointer
           dual_functional(&dual_functional),
           timestep_no(timestep_no)                 //used in output_results()
           //mpi_communicator(MPI_COMM_WORLD),
           //this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
           //base_mesh(base_mesh),
           //present_time(present_time),
           //end_time(end_time)
{
   //note, in the above constructor, the fe_dual must be initialized with FE_DGQArbitraryNodes, which 
   //is the same as in primal_solver.
}

template<int dim>
DualSolver<dim>::~DualSolver()
{
   dof_handler_dual.clear();
}

template<int dim>
DualSolver<dim>::Integrator::
Integrator(const DoFHandler<dim>& dof_handler)
          :
          dof_info(dof_handler)
{}

//prepare the system_matrix, system_rhs and solution of the dual_problem. set them to correct size.
template<int dim>
void
DualSolver<dim>::setup_system()
{
   dof_handler.distribute_dofs(fe);
   dof_handler_dual.distribute_dofs(fe_dual);
   std::cout<<"number of degrees of freedom in dual problem: "
            <<dof_handler_dual.n_dofs()
            <<std::endl;
   solution_dual.reinit(dof_handler_dual.n_dofs());
   dual_weights.reinit(dof_handler_dual.n_dofs());
   system_rhs_dual.reinit(dof_handler_dual.n_dofs());
   
//------------------------------------------------------------------------------------------------
   /*
   //return an IndexSet describing the set of locally owned DoFs as a subset of 0..n_dofs()
   IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
   //extract the set of global DoF indices that are active on the current DoFHandler. For regular DoFHandlers,
   //these are all DoF indices, but for DoFHandler objects built on parallel::distributed::Triangulation,
   //this set is the union of DoFHanler::locally_owned_dofs() and the DoF indices on all ghost cells.
   DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
   */
   /*
   const std::vector<IndexSet> locally_owned_dofs_per_proc =
                                  DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
   const IndexSet locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];

   solution_dual.reinit(locally_owned_dofs, mpi_communicator);
   system_rhs_dual.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
   */
//------------------------------------------------------------------------------------------------
   //set cell index. used to compute dt and mu_shock
   /*
   unsigned int index = 0;
   for(typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
                                                         cell != triangulation.end();
                                                         ++cell, ++index)
      cell->set_user_index(index);
   */
   DynamicSparsityPattern c_sparsity(dof_handler_dual.n_dofs());
   DoFTools::make_flux_sparsity_pattern(dof_handler_dual, c_sparsity);
   sparsity_pattern_origin.copy_from(c_sparsity);

   system_matrix_origin.reinit(sparsity_pattern_origin);   //system_matrix_dual is its transpose

   system_matrix_dual.reinit(sparsity_pattern_dual);  //this sparsity_pattern will be changed after we 
//compute system_matrix_dual   

   //generate an identity sparse matrix
   sparsity_pattern_identity.reinit(dof_handler_dual.n_dofs(),dof_handler_dual.n_dofs(), 1);
   sparsity_pattern_identity.compress();
   system_matrix_identity.reinit(sparsity_pattern_identity);  //throw error
   for(SparseMatrix<double>::size_type i =0; i<system_matrix_identity.m(); i++)
      system_matrix_identity.set(i, i, 1);


   //then interpolate the primal_solution to the dual_space, we will use it as linearization point 
   //dof_handler and dof_handler_dual must be subscribed to the same triangulation 
   primal_solution.reinit(dof_handler_dual.n_dofs());    //denote primal_solution in dual_space
   FETools::interpolate(dof_handler, 
                        primal_solution_origin,          //input
                        dof_handler_dual,
                        primal_solution);                //output
}

template<int dim>
void
DualSolver<dim>::assemble_matrix()
{
   //firstly, we initialize the integrator, this is almost the same as in ConservationLaw,
   //the only difference is: we only need to assmble the system_matrix_dual here
   //and don't need to worry about the system_rhs_dual.
   Integrator integrator(dof_handler_dual);
   //all the following quadrature formulas has n_gauss_points = fe_degree_dual+1;
   integrator.info_box.initialize_gauss_quadrature(quadrature_formula.size(),
                                                   face_quadrature_formula.size(),
                                                   face_quadrature_formula.size());

   integrator.info_box.initialize_update_flags ();
   integrator.info_box.add_update_flags_all (update_values |
                                             update_quadrature_points |
                                             update_JxW_values);
   integrator.info_box.add_update_flags_cell     (update_gradients);
   integrator.info_box.add_update_flags_boundary (update_normal_vectors | update_gradients);
   integrator.info_box.add_update_flags_face     (update_normal_vectors | update_gradients);

   integrator.info_box.initialize (fe_dual, mapping);       

   //integrator.assembler.initialize (system_matrix_dual);
   integrator.assembler.initialize (system_matrix_origin);

   //call the worker functions on cells/boundaries/faces to assemble the dual_matrix
   //system_matrix_dual = 0;
   system_matrix_origin = 0;
   std::cout<<"start to assemble the system_matrix_origin..."<<std::endl;
   MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim>>
                   (dof_handler_dual.begin_active(),
                    dof_handler_dual.end(),
                    integrator.dof_info,
                    integrator.info_box,
                    boost::bind(&DualSolver<dim>::integrate_cell_term, this, _1, _2),         //TODO
                    boost::bind(&DualSolver<dim>::integrate_boundary_term, this, _1, _2),
                    boost::bind(&DualSolver<dim>::integrate_face_term, this, _1, _2, _3, _4),
                    integrator.assembler);
   //system_matrix_dual.compress();
//=====================================================
   std::cout<<"finish assembling the system_matrix_origin"<<std::endl;
    // print out the system_matrix_origin(the transpose of the dual matrix)
   /*
   static int count = 0;
   if(count == 0){
   std::ofstream out("dual matrix",std::ios::out);
   out.precision(15);
   system_matrix_origin.print(out);
   out.close(); 
   count++;  
   }
   */
//=====================================================
}

template<int dim>
void
DualSolver<dim>::assemble_rhs()
{
   //assemble rhs of the dual_system. input: primal_solution, output: rhs of the dual_problem
   //rhs is a member of the dual class. note here for the PointValueEvaluation, primal_solution is not needed.
   dual_functional->assemble_rhs(dof_handler_dual, mapping, primal_solution, system_rhs_dual);
}

template<int dim>
std::pair<unsigned int, double>
DualSolver<dim>::solve()
{
   /*get the transpose of system_matrix_origin, store the result in system_matrix_dual
     system_matrix_origin is with sparsity_pattern_origin(with dof_handler_dual),
     system_matrix_dual is with sparsity_pattern_dual,
     system_matrix_identity is with sparsity_pattern_identity
   */
   std::cout<<"start to transpose the system_matrix_origin..."<<std::endl;
   system_matrix_origin.Tmmult(system_matrix_dual, system_matrix_identity);
   std::cout<<"finish transposing the system_matrix_origin"<<std::endl;
   /*
   //print out the system_matrix_origin
   static int count = 0;
   if(count == 0){
      std::ofstream out("dual matrix",std::ios::out);
      out.precision(15);
      system_matrix_origin.print(out);
      out.close(); 
      count++;  
   }
  //system_matrix_dual.print_pattern(std::cout);
  // static int count = 0;
   if(count == 1){
      std::ofstream out("dual matrix",std::ios::out);
      out.precision(15);
      system_matrix_dual.print(out);
      out.close(); 
      count++;  
   }
   */
   /*
   std::cout<<"system_matrix_dual dimension is "
            <<system_matrix_dual.m()<<"x"<<system_matrix_dual.n()
            <<std::endl;
   */
   //now system_matrix_dual, system_rhs_dual are all done, solve it to get solution_dual
//---------------------------------------------------------------------------------------------------
   //direct solver, the result is very good,
   //but too slow and doesn't work for very large matrix. 
   /*
   std::cout<<"start to solve the dual system..."<<std::endl;
   SparseDirectUMFPACK solver;
   solver.initialize(system_matrix_dual);
   solver.vmult(solution_dual, system_rhs_dual); 
   std::cout<<"finish solving the dual system"<<std::endl; 
   return std::pair<unsigned int, double>(1,0); 
   */
//---------------------------------------------------------------------------------------------------
   //TrilinosWrappers::SolverGMRES
   //system_matrix_dual.transpose();    

   /*
   Epetra_Vector x(View, system_matrix_dual.trilinos_matrix().DomainMap(), solution_dual.begin());
   Epetra_Vector b(View, system_matrix_dual.trilinos_matrix().RangeMap(), system_rhs_dual.begin());
 
   AztecOO solver;
   solver.SetAztecOption(AZ_output, (parameters.output == Parameters::Solver::quiet ? AZ_none: AZ_all));
   solver.SetAztecOption(AZ_solver, AZ_gmres);
   solver.SetRHS(&b);
   solver.SetLHS(&x);
   
   solver.SetAztecOption(AZ_precond,         AZ_dom_decomp);
   solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
   solver.SetAztecOption(AZ_overlap,         0);
   solver.SetAztecOption(AZ_reorder,         0);

   solver.SetAztecParam(AZ_drop,      parameters.ilut_drop);
   solver.SetAztecParam(AZ_ilut_fill, parameters.ilut_fill);
   solver.SetAztecParam(AZ_athresh,   parameters.ilut_atol);
   solver.SetAztecParam(AZ_rthresh,   parameters.ilut_rtol);

   solver.SetUserMatrix(const_cast<Epetra_CrsMatrix *> (&system_matrix_dual.trilinos_matrix()));

   solver.Iterate(parameters.max_iterations, parameters.linear_residual);

   return std::pair<unsigned int, double> (solver.NumIters(), solver.TrueResidual());
   */
//---------------------------------------------------------------------------------------------------   

   std::cout<<"start to solve the dual system..."<<std::endl;
   //used to determine whether the iteration should be continued,
   //30 is the maximum number of iteration steps before failure, default tolerance
   //to determine success of the iteration is 1e-10
   SolverControl solver_control(30);
   
   //set the number of temparary vectors to 50, i.e. do a restart every 48 iterations,
   //set right_preconditioning = true, set use_default_residual = true
   SolverGMRES<>::AdditionalData gmres_data(50);

   SolverGMRES<> solver(solver_control, gmres_data);
   
   //see PreconditionBlock, this class assumes the MatrixType consisting of invertible
   //blocks on the diagonal and provides the inversion of the diagonal blocks of the matrix
//   PreconditionBlockSSOR<TrilinosWrappers::SparseMatrix, float> preconditioner;
   //block size must be given, here is the dofs_per_cell
//   preconditioner.initialize(system_matrix_dual, fe_dual.dofs_per_cell);
   
   //TrilinosWrappers::PreconditionBlockSSOR preconditioner; 
   //preconditioner.transpose();              //throw error in this line
   /*
   PreconditionBlockSSOR<SparseMatrix<double>, float> preconditioner; 
   preconditioner.initialize(system_matrix_dual, fe_dual.dofs_per_cell);
   */
   PreconditionLU<SparseMatrix<double>, float> preconditioner;

   /*  //Jacobi preconditioner
   PreconditionBlockJacobi<SparseMatrix<double>, float> preconditioner;
   preconditioner.initialize(system_matrix_dual, 
                             PreconditionJacobi<SparseMatrix<double>>::AdditionalData(.6));
   */
   //PreconditionBlockSSOR<PETScWrappers::SparseMatrix, float> preconditioner;
   //preconditioner.initialize(system_matrix_dual, fe_dual.dofs_per_cell);
   
   try{
      solver.solve(system_matrix_dual, solution_dual, system_rhs_dual, preconditioner);
   }
   catch(...){
      std::cout<<" *** NO convergence in gmres when solving dual_problem ***\n";
   }
   std::cout<<"finish solving the dual system"<<std::endl;
   return std::pair<unsigned int, double>(solver_control.last_step(),
                                          solver_control.last_value());
   
}

template<int dim>
void
DualSolver<dim>::output_results()
{
   int dual_iter = timestep_no/parameters.refine_iter_step;
   std::string filename = "dual_solution-" + Utilities::to_string<unsigned int>(dual_iter, 4) + ".plt";  //TODO: file type
   std::ofstream output (filename.c_str());    //convert std::string(C++ style) to const char*(C style)
   //output the dual_solution
   DataOut<dim> data_out;
   data_out.attach_dof_handler (dof_handler_dual);
   data_out.add_data_vector (solution_dual,
                             EulerEquations<dim>::component_names ()
                             /*DataOut<dim>::type_dof_data*/);
   //output the dual_weights, this will be quite similar to the dual_solution, but with some difference
   std::vector<std::string> dual_weights_names;
   dual_weights_names.push_back ("XMomentum_dual_weights");
   dual_weights_names.push_back ("YMomentum_dual_weights");
   if(dim==3)
      dual_weights_names.push_back ("ZMomentum_dual_weights");
   dual_weights_names.push_back ("Density_dual_weights");
   dual_weights_names.push_back ("Energy_dual_weights");
   data_out.add_data_vector (dual_weights,
                             dual_weights_names);
   
   data_out.build_patches (mapping, fe_degree_dual);

   std::cout << "Writing file of dual_solution" << std::endl;   
   data_out.write_tecplot (output);
}

//return a const reference, so that other objects can read DualSolver's private member variable. 
template<int dim>
void
DualSolver<dim>::compute_DWR_indicators(Vector<double>& refinement_indicators)
{
   Assert (refinement_indicators.size()==triangulation->n_active_cells(),
           ExcDimensionMismatch(refinement_indicators.size(), triangulation->n_global_active_cells()));
   //solve the dual_problem to get the dual_weights
   setup_system();
   assemble_matrix();
//======================================================
   /*
   static int count = 0;
   if(count == 0){
   std::ofstream out("dual matrix",std::ios::out);
   out.precision(15);
   system_matrix_origin.print(out);
   out.close(); 
   count++;  
   }
   */
//======================================================
   assemble_rhs();

   std::pair<unsigned int, double> convergence = solve();

   std::cout<<"finish solving dual_problem ... "
            <<std::endl
            <<convergence.first<<" "<<convergence.second
            <<std::endl;
   
   //then in the following, compute the error indicator using dual_weights and residual
   //------------------------compute the dual_weights---------------------------------
   //dual_weights: z-z_h. z is in dual_space, z_h is z's interpolation into primal_space. 
   //consider this: why we must use z-z_h to denote the dual_weights, rather using z itself?
   //follow function will compute (Id-I_h)z for a given dof-function z, where I_h is the 
   //interpolation from fe1 to fe2, i.e. from dual_space to primal_space
   FETools::interpolation_difference (dof_handler_dual,  //dof1
                                      solution_dual,     //z
                                      fe,                //fe2
                                      dual_weights);     //z_differnce
   
   dual_weights = solution_dual;
   std::cout<<"start to compute the dual_weights..."<<std::endl;
   //-----------------------prepare all the data_structure----------------------------
   FEValues<dim> fe_values(fe_dual, quadrature_formula, update_values|          //TODO: quadrature_formula
                                                        update_gradients|
                                                        update_quadrature_points|
                                                        update_JxW_values);
   const unsigned int n_q_points = quadrature_formula.size();
   const unsigned int n_components  = EulerEquations<dim>::n_components;
   const unsigned int dofs_per_cell = fe_dual.dofs_per_cell;
   std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
   std::vector<types::global_dof_index> dof_indices_neighbor(dofs_per_cell);
   //data structures used to integrate on cells
   std::vector<Vector<double>> dual_weights_cell_values(n_q_points, Vector<double>(n_components));
   std::vector<Vector<double>> cell_residual(n_q_points, Vector<double>(n_components));
   //std::vector<Vector<double>> W(n_q_points, Vector<double>(n_components));
   //std::vector<std::vector<Vector<double>>> grad_W(n_q_points, std::vector<Vector<double>>(dim, Vector<double>(n_components)));
   Table<2, double> W(n_q_points, n_components);
   Table<3, double> grad_W(n_q_points, n_components, dim);
   //note: why we use array rather than FullMatrix? sice FullMatrix need arguments to allocate memory each
   //time such a matrix is created.
   //std::vector<FullMatrix<double>> A_x(n_q_points, FullMatrix<double>(n_components,n_components)),
   //                                A_y(n_q_points, FullMatrix<double>(n_components,n_components));
   std::vector<std::array<std::array<double, n_components>, n_components>> A_x(n_q_points),
                                                                           A_y(n_q_points);
   
   std::vector<double> mu_shock_cell(n_q_points);
   std::vector<double> residual_average(n_q_points);
   std::vector<std::vector<Vector<double>>> grad_dual_weights(n_q_points, std::vector<Vector<double>>(dim, Vector<double>(n_components)));
   /*std::vector<std::vector<Tensor<1, dim>>> grad_dual_weights(n_q_points);
   for(unsigned int p=0; p!=n_q_points; ++p)
      grad_dual_weights[p].resize(n_components);   //denote d(z-z_h)/dx and d(z-z_h)/dy
   */
   FEFaceValues<dim> fe_face_values_cell(fe_dual, face_quadrature_formula, update_values|
                                                                           update_quadrature_points|
                                                                           update_gradients|
                                                                           update_JxW_values|
                                                                           update_normal_vectors),
                     fe_face_values_neighbor(fe_dual, face_quadrature_formula, update_values|
                                                                               update_quadrature_points|
                                                                               update_gradients|
                                                                               update_JxW_values|
                                                                               update_normal_vectors);
   FESubfaceValues<dim> fe_subface_values_cell(fe_dual, face_quadrature_formula, update_values|
                                                                                 update_quadrature_points|
                                                                                 update_JxW_values|
                                                                                 update_normal_vectors),
                        fe_subface_values_neighbor(fe_dual,face_quadrature_formula,update_values|
                                                                                   update_quadrature_points);
   const unsigned int n_face_q_points = face_quadrature_formula.size();
   //data structures used to integrate on faces
   std::vector<Vector<double>> dual_weights_face_values(n_face_q_points, Vector<double>(n_components));
   std::vector<Vector<double>> face_residual(n_face_q_points, Vector<double>(n_components));
   //std::vector<Vector<double>> Wplus(n_face_q_points, Vector<double>(n_components));
   //std::vector<Vector<double>> Wminus(n_face_q_points, Vector<double>(n_components));
   Table<2, double> Wplus(n_face_q_points, n_components),
                    Wminus(n_face_q_points, n_components);
   std::vector<Vector<double>> boundary_values(n_face_q_points, Vector<double>(n_components));
   
   //std::vector<Vector<double>> normal_flux(n_face_q_points, Vector<double>(n_components)); 
   //std::vector<Vector<double>> numerical_flux(n_face_q_points, Vector<double>(n_components));
   std::vector<std::array<double, n_components>> normal_flux(n_face_q_points); //should it be an array ?
   std::vector<std::array<double, n_components>> numerical_flux(n_face_q_points);

   //define a map from face_iterator to the face_term_error
   //typename std::map<typename DoFHandler<dim>::face_iterator, Vector<double>> face_integrals;
   typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_dual.begin_active(),
                                                  endc = dof_handler_dual.end();
   /*for(; cell!=endc; ++cell){
      for(unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no){
         face_integrals[cell->face(face_no)].reinit(dim);  //TODO: what mean?
         face_integrals[cell->face(face_no)] = -1e20;
      }
   }*/
   //----------------------------------loop-----------------------------------------
   //cell = dof_handler_dual.begin_active(); 
   unsigned int present_cell = 0;
   for(; cell!=endc; ++cell, ++present_cell){
      //-------------------------integrate_over_cell----------------------------
      /*
      std::cout<<"present_cell is "<<present_cell<<std::endl;
      if(present_cell > 2694){
         std::cout<<"the curent element is 2695"<<std::endl;
         std::cout<<" ---------------------------------  "<<std::endl;
      }
      */ 
      fe_values.reinit(cell);
      cell->get_dof_indices (dof_indices);  //store global_dof_indices corresponding to local_dof_indices
      fe_values.get_function_values(dual_weights, dual_weights_cell_values);
      //set initial values to 0 in every loop 
      for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
         residual_average[q_point] = 0;
         for(unsigned int c=0; c<n_components; ++c){
            W[q_point][c] = 0;               
            cell_residual[q_point][c] = 0;
            for(unsigned int d=0; d<dim; d++)
               grad_W[q_point][c][d] = 0;         //dW/dx and dW/dy at quadrature_point, they are Vectors
         }
         for(unsigned int d=0; d<dim; d++)
            grad_dual_weights[q_point][d] = 0;
      }
      
      for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
         //get grad_W and Jacobi matrix on all quadrature points
         for(unsigned int i=0; i<dofs_per_cell; ++i){
            const unsigned int c = fe_values.get_fe().system_to_component_index(i).first;
            W[q_point][c] += primal_solution(dof_indices[i]) * 
                             fe_values.shape_value_component(i,q_point,c);
            for(unsigned int d=0; d<dim; d++)
               grad_W[q_point][c][d] += primal_solution(dof_indices[i]) * 
                                        fe_values.shape_grad_component(i,q_point,c)[d];
         }
         EulerEquations<dim>::compute_Jacobi_matrix (W[q_point], A_x[q_point], A_y[q_point]); //get A_x, A_y

         //use Jacobi matrix and grad_W to compute A_x*dW/dx+A_y*dW/dy, i.e. the residual
         /*A_x[q_point].vmult_add(cell_residual[q_point], grad_W[q_point][0]);  //cell_residual+=A_x*dW/dx+A_y*dW/dy
         A_y[q_point].vmult_add(cell_residual[q_point], grad_W[q_point][1]);
         */
         for(unsigned int row=0; row<n_components; row++){
            for(unsigned int col=0; col<n_components; col++)
            cell_residual[q_point][row] += A_x[q_point][row][col] * grad_W[q_point][col][0] +
                                           A_y[q_point][row][col] * grad_W[q_point][col][1];
                                                       
         }
      }
      //cell_residuals are prepared, dual_weightes are prepared, now compute the integral
      double cell_integral = 0;
      for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
         cell_integral += cell_residual[q_point] * 
                          dual_weights_cell_values[q_point] *        //here is Vector*Vector
                          fe_values.JxW(q_point);  
      }
      refinement_indicators(present_cell) -= cell_integral;
      //compute mu_shock_cell, it's used to compute the integral term related to diffusion 
      for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
         for(unsigned int c=0; c<n_components; ++c)
            residual_average[q_point] += std::abs(cell_residual[q_point][c]); 
         residual_average[q_point] /= n_components;
         mu_shock_cell[q_point] = parameters.diffusion_coef*
                                std::pow(fe_values.get_cell()->diameter(), 2-parameters.diffusion_power)*
                                residual_average[q_point];
      }
      //integrate the cell term related to diffusion
      for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
         for(unsigned int i=0; i<dofs_per_cell; ++i){
            const unsigned int c = fe_values.get_fe().system_to_component_index(i).first;
            for(unsigned int d=0; d<dim; d++)
               grad_dual_weights[q_point][d][c] += dual_weights(dof_indices[i]) * 
                                                   fe_values.shape_grad_component(i,q_point,c)[d];
         }
      }
      //fe_values.get_function_gradients(dual_weights, grad_dual_weights);
      cell_integral = 0;
      for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
         for(unsigned int d=0; d<dim; ++d)
            for(unsigned int c=0; c<n_components; ++c)
            cell_integral += mu_shock_cell[q_point] * 
                             grad_W[q_point][c][d] * grad_dual_weights[q_point][d][c] *  
                             fe_values.JxW(q_point);
      }
      refinement_indicators(present_cell) -= cell_integral;
      //-------------------------integrate_over_face----------------------------
      for(unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no){
         //set data structure's initial value to 0
         for(unsigned int q=0; q<n_face_q_points; ++q)
            for(unsigned int c=0; c<n_components; ++c){
               Wplus[q][c] = 0;
               Wminus[q][c]= 0;
            }
      
         //if the face at boundary
         if(cell->face(face_no)->at_boundary()){
            const unsigned int& boundary_id = cell->face(face_no)->boundary_id(); 
            //get Wplus at all quadrature points on current face
            fe_face_values_cell.reinit(cell, face_no); 
            for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
               for(unsigned int i=0; i<dofs_per_cell; ++i){
                  const unsigned int c = fe_face_values_cell.get_fe().system_to_component_index(i).first;
                  Wplus[q_point][c] += primal_solution(dof_indices[i]) * 
                                       fe_face_values_cell.shape_value_component(i,q_point,c);
               }
            }
            //fe_face_values_cell.get_function_values(primal_solution, Wplus);
            //get Wminus at all quadrature points on current face
            parameters.boundary_conditions[boundary_id].values.
                       vector_value_list(fe_face_values_cell.get_quadrature_points(),boundary_values);
            typename EulerEquations<dim>::BoundaryKind boundary_kind = 
                                                       parameters.boundary_conditions[boundary_id].kind;
            for(unsigned int p=0; p<n_face_q_points; p++){
               EulerEquations<dim>::compute_Wminus(boundary_kind,
                                                   fe_face_values_cell.normal_vector(p),
                                                   Wplus[p],
                                                   boundary_values[p],
                                                   Wminus[p]);
            }
            //now Wplus and Wminus are known, use them to compute normal_flux and numerical_normal_flux
            for(unsigned int p=0; p<n_face_q_points; p++){
               //compute the normal_flux and numerical_flux on current quadrature_point
               EulerEquations<dim>::compute_normal_flux(Wplus[p], 
                                                        fe_face_values_cell.normal_vector(p),
                                                        normal_flux[p]);
               numerical_normal_flux(fe_face_values_cell.normal_vector(p),
                                     Wplus[p],
                                     Wminus[p],
                                     numerical_flux[p]);
               //compute the face_residual on current quadrature_point
               for(unsigned int c=0; c<n_components; c++)
                  face_residual[p][c] = normal_flux[p][c] - numerical_flux[p][c];  
            }
            //get the dual_weights
            fe_face_values_cell.get_function_values(dual_weights, dual_weights_face_values);
            //compute the face integral,i.e. the face_term_error
            double face_integral = 0;
            for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
               face_integral += face_residual[q_point] * 
                                dual_weights_face_values[q_point] *        //here is Vector*Vector
                                fe_face_values_cell.JxW(q_point);  
            }
            refinement_indicators(present_cell) += face_integral;
         }
         //if the neighbor is finer(i.e. the current face has children)
         else if(cell->face(face_no)->has_children()){
            //
            const unsigned int neighbor_neighbor = cell->neighbor_of_neighbor(face_no);
            const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);
            for(unsigned int subface_no=0; subface_no<face->n_children(); ++subface_no){
               //get Wplus. note that here Wplus only contains information on current subface
               fe_subface_values_cell.reinit(cell, face_no, subface_no);
               //fe_subface_values_cell.get_function_values(primal_solution, Wplus);
               for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
                  for(unsigned int i=0; i<dofs_per_cell; ++i){
                     const unsigned int c = fe_subface_values_cell.get_fe().
                                            system_to_component_index(i).first;
                     Wplus[q_point][c] += primal_solution(dof_indices[i]) * 
                                          fe_subface_values_cell.shape_value_component(i,q_point,c);
                  }
               }
               //get Wminus
               const typename DoFHandler<dim>::active_cell_iterator neighbor_child = 
                                          cell->neighbor_child_on_subface(face_no, subface_no);
               Assert(neighbor_child->face(neighbor_neighbor)==cell->face(face_no)->child(subface_no),
                      ExcInternalError());
               fe_face_values_neighbor.reinit(neighbor_child, neighbor_neighbor);
               neighbor_child->get_dof_indices(dof_indices_neighbor);
               //fe_face_values_neighbor.get_function_values(primal_solution, Wminus);
                for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
                  for(unsigned int i=0; i<dofs_per_cell; ++i){
                     const unsigned int c = fe_face_values_neighbor.get_fe().
                                            system_to_component_index(i).first;
                     Wminus[q_point][c] += primal_solution(dof_indices_neighbor[i]) * 
                                           fe_face_values_neighbor.shape_value_component(i,q_point,c);
                  }
               }
               //use Wplus and Wminus on current subface to compute the flux and numerical flux
               for(unsigned int p=0; p<n_face_q_points; p++){
                  //compute the normal_flux and numerical_flux on current quadrature_point
                  EulerEquations<dim>::compute_normal_flux(Wplus[p], 
                                                           fe_subface_values_cell.normal_vector(p),
                                                           normal_flux[p]);
                  numerical_normal_flux(fe_subface_values_cell.normal_vector(p),
                                        Wplus[p],
                                        Wminus[p],
                                        numerical_flux[p]);
                  //compute the face_residual on current quadrature_point
                  for(unsigned int c=0; c<n_components; c++)
                     face_residual[p][c] = normal_flux[p][c] - numerical_flux[p][c];  
               }
               //get the dual_weights
               fe_subface_values_cell.get_function_values(dual_weights, dual_weights_face_values);
               //compute the face integral,i.e. the face_term_error
               double subface_integral = 0;
               for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
                  subface_integral += face_residual[q_point] * 
                                      dual_weights_face_values[q_point] *        //here is Vector*Vector
                                      fe_subface_values_cell.JxW(q_point);  
               }
               refinement_indicators(present_cell) += subface_integral;
            }//end subface loop
         }
         //if the neighbor is as fine as our cell
         else if(!cell->neighbor_is_coarser(face_no)){
            //get Wplus
            fe_face_values_cell.reinit(cell, face_no);
            //fe_face_values_cell.get_function_values(primal_solution, Wplus);
             for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
                  for(unsigned int i=0; i<dofs_per_cell; ++i){
                     const unsigned int c = fe_face_values_cell.get_fe().
                                            system_to_component_index(i).first;
                     Wplus[q_point][c] += primal_solution(dof_indices[i]) * 
                                          fe_face_values_cell.shape_value_component(i,q_point,c);
                  }
             }
            //get Wminus
            Assert(cell->neighbor(face_no).state() == IteratorState::valid, ExcInternalError()); //check if the neighbor exists
            const unsigned int neighbor_neighbor = cell->neighbor_of_neighbor(face_no);
            const typename DoFHandler<dim>::active_cell_iterator neighbor = cell->neighbor(face_no);
            fe_face_values_neighbor.reinit(neighbor, neighbor_neighbor);
            neighbor->get_dof_indices(dof_indices_neighbor);
            //fe_face_values_neighbor.get_function_values(primal_solution, Wminus);
            for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
                  for(unsigned int i=0; i<dofs_per_cell; ++i){
                     const unsigned int c = fe_face_values_neighbor.get_fe().
                                            system_to_component_index(i).first;
                     Wminus[q_point][c] += primal_solution(dof_indices_neighbor[i]) * 
                                           fe_face_values_neighbor.shape_value_component(i,q_point,c);
                  }
             }
            //use Wplus and Wminus on current subface to compute the flux and numerical flux
            for(unsigned int p=0; p<n_face_q_points; p++){
               EulerEquations<dim>::compute_normal_flux(Wplus[p], 
                                                        fe_face_values_cell.normal_vector(p),
                                                        normal_flux[p]);
               numerical_normal_flux(fe_face_values_cell.normal_vector(p),
                                     Wplus[p],
                                     Wminus[p],
                                     numerical_flux[p]);
               //compute the face_residual on current quadrature_point
               for(unsigned int c=0; c<n_components; c++)
                  face_residual[p][c] = normal_flux[p][c] - numerical_flux[p][c];  
            }
            //get the dual_weights
            fe_face_values_cell.get_function_values(dual_weights, dual_weights_face_values);
            //compute the face integral,i.e. the face_term_error
            double face_integral = 0;
            for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
               face_integral += face_residual[q_point] * 
                                dual_weights_face_values[q_point] *        //here is Vector*Vector
                                fe_face_values_cell.JxW(q_point);  
            }
            refinement_indicators(present_cell) += face_integral;  
         }
         //if the neighbor is coarser
         else{
            //get Wplus
            fe_face_values_cell.reinit(cell, face_no);
            //fe_face_values_cell.get_function_values(primal_solution, Wplus);  
            for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
                  for(unsigned int i=0; i<dofs_per_cell; ++i){
                     const unsigned int c = fe_face_values_cell.get_fe().
                                            system_to_component_index(i).first;
                     Wplus[q_point][c] += primal_solution(dof_indices[i]) * 
                                          fe_face_values_cell.shape_value_component(i,q_point,c);
                  }
             }          
            Assert(cell->neighbor(face_no).state() == IteratorState::valid, ExcInternalError()); 
            Assert(cell->neighbor_is_coarser(face_no), ExcInternalError());
            //get Wminus
            const unsigned int neighbor_neighbor = cell->neighbor_of_coarser_neighbor(face_no).first;
            const unsigned int subface_neighbor = cell->neighbor_of_coarser_neighbor(face_no).second;
            const typename DoFHandler<dim>::active_cell_iterator neighbor = cell->neighbor(face_no);
            fe_subface_values_neighbor.reinit(neighbor, neighbor_neighbor, subface_neighbor);
            neighbor->get_dof_indices(dof_indices_neighbor);
            //fe_subface_values_neighbor.get_function_values(primal_solution, Wminus);
            for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
                  for(unsigned int i=0; i<dofs_per_cell; ++i){
                     const unsigned int c = fe_subface_values_neighbor.get_fe().
                                            system_to_component_index(i).first;
                     Wminus[q_point][c] += primal_solution(dof_indices_neighbor[i]) * 
                                           fe_subface_values_neighbor.shape_value_component(i,q_point,c);
                  }
             }
            //use Wplus and Wminus on current subface to compute the flux and numerical flux
            for(unsigned int p=0; p<n_face_q_points; p++){
               EulerEquations<dim>::compute_normal_flux(Wplus[p], 
                                                        fe_face_values_cell.normal_vector(p),
                                                        normal_flux[p]);
               numerical_normal_flux(fe_face_values_cell.normal_vector(p),
                                     Wplus[p],
                                     Wminus[p],
                                     numerical_flux[p]);
               //compute the face_residual on current quadrature_point
               for(unsigned int c=0; c<n_components; c++)
                  face_residual[p][c] = normal_flux[p][c] - numerical_flux[p][c];  //here is Vector-Vector
            }
            //get the dual_weights
            fe_face_values_cell.get_function_values(dual_weights, dual_weights_face_values);
            //compute the face integral,i.e. the face_term_error
            double face_integral = 0;
            for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
               face_integral += face_residual[q_point] * 
                                dual_weights_face_values[q_point] *        //here is Vector*Vector
                                fe_face_values_cell.JxW(q_point);  
            }
            refinement_indicators(present_cell) += face_integral;  
         }
      }//end face_loop  
 
   }// end cell_loop
   std::cout<<"finish computing the error indicator"<<std::endl;
   std::cout<<"estimated error = "
            <<std::accumulate(refinement_indicators.begin(), refinement_indicators.end(), 0.)
            <<std::endl;

   //set the indicators on each cell to positive
   for(Vector<double>::iterator i = refinement_indicators.begin(); i!=refinement_indicators.end(); ++i)
      *i = std::fabs(*i);

   //output the dual_solution
   output_results();  
   
   std::cout<<"start to write out the refinement_indicator..."<<std::endl;
   //std::cout<<refinement_indicators[2695]<<std::endl;
   //refinement_indicators.print(std::cout, 4);
   DataOut<dim> data_out;
   data_out.attach_dof_handler (dof_handler);
   data_out.add_data_vector (refinement_indicators, 
                             "refinement_indicator",
                             DataOut<dim>::type_cell_data);//TODO

   data_out.build_patches (mapping);

   std::cout << "Writing file of refinement_indicator" << std::endl;   
   std::ofstream output ("refinement_indicator.plt");
   data_out.write_tecplot (output);
}

//
template<int dim>
void
DualSolver<dim>::integrate_cell_term(DoFInfo& dinfo, CellInfo& info)
{
   FullMatrix<double>& local_matrix = dinfo.matrix(0).matrix;
   //Vector<double>& local_vector   = dinfo.vector(0).block(0);
   std::vector<unsigned int>& dof_indices = dinfo.indices;

   const FEValuesBase<dim>& fe_v    = info.fe_values();
   const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
   const unsigned int n_q_points    = fe_v.n_quadrature_points;

   //const double time_step = dt(cell_number(dinfo.cell));
   //Table<2,Sacado::Fad::DFad<double> >
   //   dWdt (n_q_points, EulerEquations<dim>::n_components);
   
   Table<2, Sacado::Fad::DFad<double>>
      W (n_q_points, EulerEquations<dim>::n_components);
   
   Table<3, Sacado::Fad::DFad<double>> 
      grad_W (n_q_points, EulerEquations<dim>::n_components, dim);
   
   // extract primal_solution and store as local variable. here primal_solution is used.
   std::vector<Sacado::Fad::DFad<double> > independent_local_dof_values(dofs_per_cell);
   for (unsigned int i=0; i<dofs_per_cell; ++i)
      independent_local_dof_values[i] = primal_solution(dof_indices[i]);
   
   for (unsigned int i=0; i<dofs_per_cell; ++i)
      independent_local_dof_values[i].diff (i, dofs_per_cell);  
   
   for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
         const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;
         
         Sacado::Fad::DFad<double> dof =  independent_local_dof_values[i];

         W[q][c] += dof * fe_v.shape_value_component(i, q, c);
         
         for (unsigned int d = 0; d < dim; d++)
            grad_W[q][c][d] += dof * fe_v.shape_grad_component(i, q, c)[d];
         
      }  
   /*
   typedef Sacado::Fad::DFad<double> FluxMatrix[EulerEquations<dim>::n_components][dim];
   FluxMatrix *flux = new FluxMatrix[n_q_points];
   
   typedef Sacado::Fad::DFad<double> ForcingVector[EulerEquations<dim>::n_components];
   ForcingVector *forcing = new ForcingVector[n_q_points];
   */
   std::vector<std::array<std::array<Sacado::Fad::DFad<double>, dim>, EulerEquations<dim>::n_components>> 
                                                                       flux(n_q_points);
   std::vector<std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components>>   
                                                                    forcing(n_q_points);
//--------------------------------------------------------------------------------------
//I add following variables in order to compute the diffusion terms
//-------------------------------------------------------------------------------------- 
   //R denotes the residual vectors on quadrature points of current cell
   Table<2,Sacado::Fad::DFad<double>> R(n_q_points, EulerEquations<dim>::n_components);

   //get the jacobi on quadrature points
   /*
   typedef Sacado::Fad::DFad<double> JacobiMatrix[EulerEquations<dim>::n_components][EulerEquations<dim>::n_components];
   JacobiMatrix *A_x = new JacobiMatrix[n_q_points];
   JacobiMatrix *A_y = new JacobiMatrix[n_q_points];
   */
   std::vector<std::array<std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components>,
                                                                EulerEquations<dim>::n_components>> 
                                                  A_x(n_q_points),
                                                  A_y(n_q_points);
//--------------------------------------------------------------------------------------
   for (unsigned int q=0; q<n_q_points; ++q)
   {
      EulerEquations<dim>::compute_flux_matrix (W[q], flux[q]);
      EulerEquations<dim>::compute_forcing_vector (W[q], forcing[q]);
      EulerEquations<dim>::compute_Jacobi_matrix (W[q], A_x[q], A_y[q]); //get A_x & A_y
   }
   
   
   // Get current cell number
   //const unsigned int cell_no = cell_number (dinfo.cell);

//--------------------------------------------------------------------------------------
   // I add this vector to denote the diffusion coefficient   
   std::vector<Sacado::Fad::DFad<double>> mu_shock_cell(n_q_points);
   std::vector<Sacado::Fad::DFad<double>> R_average(n_q_points);

   for(unsigned int point=0; point<n_q_points; ++point){
     for(unsigned int c=0; c<EulerEquations<dim>::n_components; c++){
       //R[point][c] += dWdt[point][c];
       for(unsigned int j=0; j<EulerEquations<dim>::n_components; j++){   //the j-th component of the variable vector W
         R[point][c] += A_x[point][c][j]*grad_W[point][j][0];
         R[point][c] += A_y[point][c][j]*grad_W[point][j][1];  
       }
     }
   }//finish computing Residual vector R(point, components)

   for(unsigned int point=0; point<n_q_points; ++point){
     for(unsigned int c=0; c<EulerEquations<dim>::n_components; ++c){
       R_average[point] += std::abs(R[point][c]);  //sum all the components of R on current point 
     } 
     R_average[point] /= EulerEquations<dim>::n_components;
     mu_shock_cell[point] = parameters.diffusion_coef*
                            std::pow(fe_v.get_cell()->diameter(),2-parameters.diffusion_power)*
                            R_average[point];
   }
   /*
   //compute the average of mu_shock_cell on a cell just for visualization
   Sacado::Fad::DFad<double> mu_shock_ave = 0;
   for(unsigned int point=0; point<n_q_points; ++point)
     mu_shock_ave += mu_shock_cell[point];
   mu_shock_ave /= n_q_points;
   mu_shock(cell_no) = mu_shock_ave.val();   */
//--------------------------------------------------------------------------------------   

   for (unsigned int i=0; i<dofs_per_cell; ++i)
   {
      Sacado::Fad::DFad<double> F_i = 0;
      
      const unsigned int
      component_i = fe_v.get_fe().system_to_component_index(i).first;
      
      // The residual for each row (i) will be accumulating
      // into this fad variable.  At the end of the assembly
      // for this row, we will query for the sensitivities
      // to this variable and add them into the Jacobian.
      
      for (unsigned int point=0; point<n_q_points; ++point)
      {        
         for (unsigned int d=0; d<dim; d++)
            F_i -= flux[point][component_i][d] *
                   fe_v.shape_grad_component(i, point, component_i)[d] *
                   fe_v.JxW(point);
         
         // Diffusion term for shocks
         /*
         if(parameters.diffusion_coef > 0.0)
            for (unsigned int d=0; d<dim; d++)
               F_i += mu_shock(cell_no) *
                      grad_W[point][component_i][d] *
                      fe_v.shape_grad_component(i, point, component_i)[d] *
                      fe_v.JxW(point);
         */
         //my new way of adding diffusiton terms
         if(parameters.diffusion_coef > 0.0)
            for (unsigned int d=0; d<dim; d++)
              F_i += mu_shock_cell[point] * 
                     grad_W[point][component_i][d] *
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.JxW(point);
         /*
         F_i -= parameters.gravity *
                forcing[point][component_i] *
                fe_v.shape_value_component(i, point, component_i) *
                fe_v.JxW(point);
         */
      }
      
      for (unsigned int k=0; k<dofs_per_cell; ++k)
         local_matrix (i, k) += F_i.fastAccessDx(k);
      
      /* used to test if I can assemble the system_matrix in transposed order, but failed  
      for (unsigned int k=0; k<dofs_per_cell; ++k)
         local_matrix (k, i) += F_i.fastAccessDx(k);
      */
      //local_vector (i) -= F_i.val();
      //local_vector.print(std::cout,3);
   }
   
   //delete[] forcing;
   //delete[] flux;
   //delete[] A_x;
   //delete[] A_y;
}
template<int dim>
void
DualSolver<dim>::integrate_boundary_term(DoFInfo& dinfo, CellInfo& info)
{
   FullMatrix<double>& local_matrix = dinfo.matrix(0).matrix;
   //Vector<double>& local_vector   = dinfo.vector(0).block(0);
   std::vector<unsigned int>& dof_indices = dinfo.indices;
   const unsigned int& face_no = dinfo.face_number;
   const double& face_diameter = dinfo.face->diameter();
   const unsigned int& boundary_id = dinfo.face->boundary_id();
   
   const FEValuesBase<dim>& fe_v = info.fe_values();
   const unsigned int n_q_points = fe_v.n_quadrature_points;
   const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
   
   std::vector<Sacado::Fad::DFad<double> >
   independent_local_dof_values (dofs_per_cell);
   
   const unsigned int n_independent_variables = dofs_per_cell;
   
   // Get cell number of two cells adjacent to this face
   //const unsigned int cell_no = cell_number (dinfo.cell);
   
   for (unsigned int i = 0; i < dofs_per_cell; i++)
   {
      independent_local_dof_values[i] = primal_solution(dof_indices[i]);
      independent_local_dof_values[i].diff(i, n_independent_variables);
   }
   
   // Conservative variable value at face
   Table<2,Sacado::Fad::DFad<double> >
   Wplus (n_q_points, EulerEquations<dim>::n_components),
   Wminus (n_q_points, EulerEquations<dim>::n_components);

   /* TODO:ADIFF
   Table<3,Sacado::Fad::DFad<double> >
   grad_Wplus  (n_q_points, EulerEquations<dim>::n_components, dim);
   */

   for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
         const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;
         Sacado::Fad::DFad<double> dof = independent_local_dof_values[i];
                                          
         Wplus[q][c] += dof * fe_v.shape_value_component(i, q, c);

         /* TODO:ADIFF
         for (unsigned int d = 0; d < dim; d++)
            grad_Wplus[q][c][d] += dof *
                                   fe_v.shape_grad_component(i, q, c)[d];
         */
      }
   
   // On the other side of face, we have boundary. We get Wminus from
   // boundary conditions

   Assert (boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
           ExcIndexRange (boundary_id, 0,
                          Parameters::AllParameters<dim>::max_n_boundaries));
   
   std::vector<Vector<double> >
   boundary_values(n_q_points, Vector<double>(EulerEquations<dim>::n_components));
   parameters.boundary_conditions[boundary_id]
   .values.vector_value_list(fe_v.get_quadrature_points(),
                             boundary_values);
   
   
   typename EulerEquations<dim>::BoundaryKind boundary_kind = 
      parameters.boundary_conditions[boundary_id].kind;
//=================================================================================================
/* //the old method 
   for (unsigned int q = 0; q < n_q_points; q++)
      EulerEquations<dim>::compute_Wminus (boundary_kind,
                                           fe_v.normal_vector(q),
                                           Wplus[q],
                                           boundary_values[q],
                                           Wminus[q]);
   
   
   // Compute numerical flux at all quadrature points
   std::vector<std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components>>
                                                               normal_fluxes(n_q_points);

   for (unsigned int q=0; q<n_q_points; ++q){
      numerical_normal_flux(fe_v.normal_vector(q),
                            Wplus[q], 
                            Wminus[q],
                            //cell_average[cell_no],
                            //cell_average[cell_no],
                            normal_fluxes[q]);
      //std::cout<<"flux of cell: "<<cell_no<<" point: "<<q<<"is ";
      //for(unsigned int i=0;i<EulerEquations<dim>::n_components;++i)
      //   std::cout<<normal_fluxes[q][i].val()<<",";
      //std::cout<<std::endl;
      }
*/
//==================================================================================================
   //the new method
   //In order to obtain an adjoint-consistent discretization, there will be 2 differences 
   //when computing the boundary flux comparing to the standard method of dg code.
   //Firstly, on the boundary, we cannot use the same numerical flux as in the interior
   //of the domain, rather, we should use H(u+,u-(u+),n) = n dot F(u-(u+)).
   //Secondly, when compute u-(u+) on solid wall boundary, we should make it satisfy
   //v dot n = 0, i.e. let v- = v - (vdotn)n, rather than letting v- = v - 2(vdotn)n. 
   //see Hartmann's paper "derivation of an adjoint consistent discontinuous galerkin
   //discretization of the compressible euler equations" for more detail.      
   for(unsigned int q = 0; q < n_q_points; q++)
      EulerEquations<dim>::compute_W_prescribed(boundary_kind,
                                                fe_v.normal_vector(q),
                                                Wplus[q],
                                                boundary_values[q],
                                                Wminus[q]);
   std::vector<std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components>>
                                                               normal_fluxes(n_q_points);
   //on all boundaries, the flux is H(u+,u-(u+),n)=n dot F(u-u(u+)), 
   //while u-(u+) varies on different boundaries kinds.
   for (unsigned int q=0; q<n_q_points; ++q)
      EulerEquations<dim>::compute_normal_flux(Wminus[q], 
                                               fe_v.normal_vector(q),
                                               normal_fluxes[q]);
//==================================================================================================
   // Now assemble the face term
   for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
      if (fe_v.get_fe().has_support_on_face(i, face_no) == true) // comment this for TODO:ADIFF
      {
         Sacado::Fad::DFad<double> F_i = 0;
         
         for (unsigned int point=0; point<n_q_points; ++point)
         {
            const unsigned int
            component_i = fe_v.get_fe().system_to_component_index(i).first;
           // std::cout<<"the normal_flux of component on current point is: "
           //          <<normal_fluxes[point][component_i]
           //          <<std::endl
           //          <<"the shape_value_component  is "
           //          <<fe_v.shape_value_component(i, point, component_i)
           //          <<"the JxW is "
           //          <<fe_v.JxW(point)
           //          <<std::endl;
            F_i += normal_fluxes[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);

            /* TODO:ADIFF
            F_i += 5.0 * mu_shock(cell_no) / dinfo.cell->diameter() *
                   (Wplus[point][component_i] - Wminus[point][component_i]) *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);

            for(unsigned int d=0; d<dim; ++d)
               F_i -= mu_shock(cell_no)         * grad_Wplus[point][component_i][d] *
                     fe_v.normal_vector(point)[d] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point)
                     +
                     mu_shock(cell_no) * 
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.normal_vector(point)[d] *
                     (Wplus[point][component_i] - Wminus[point][component_i]) *
                     fe_v.JxW(point);
                     */
         }
         
         for (unsigned int k=0; k<dofs_per_cell; ++k)
            local_matrix (i,k) += F_i.fastAccessDx(k);
         
         /* used to test if I can assemble the system_matrix in transposed order, but failed 
         for (unsigned int k=0; k<dofs_per_cell; ++k)
            local_matrix (k, i) += F_i.fastAccessDx(k);
         */
         //local_vector (i) -= F_i.val();
         //local_vector.print(std::cout,3);     //TO SHOW THE LOCAL_VECTOR
         //std::cout<<"this is boundary_term on cell: "<<cell_no<<F_i.val()<<std::endl;
      }
}
template<int dim>
void
DualSolver<dim>::integrate_face_term(DoFInfo& dinfo1, DoFInfo& dinfo2,
                                     CellInfo& info1, CellInfo& info2)
{
   FullMatrix<double>& local_matrix11 = dinfo1.matrix(0,false).matrix;
   FullMatrix<double>& local_matrix12 = dinfo1.matrix(0,true).matrix;
   //Vector<double>& local_vector   = dinfo1.vector(0).block(0);
   std::vector<unsigned int>& dof_indices = dinfo1.indices;
   const unsigned int& face_no = dinfo1.face_number;
   const double& face_diameter = dinfo1.face->diameter();
   
   FullMatrix<double>& local_matrix_neighbor22 = dinfo2.matrix(0,false).matrix;
   FullMatrix<double>& local_matrix_neighbor21 = dinfo2.matrix(0,true).matrix;
   //Vector<double>& local_vector_neighbor   = dinfo2.vector(0).block(0);
   std::vector<unsigned int>& dof_indices_neighbor = dinfo2.indices;
   const unsigned int& face_no_neighbor = dinfo2.face_number;

   const FEValuesBase<dim>& fe_v = info1.fe_values();
   const unsigned int n_q_points = fe_v.n_quadrature_points;
   const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
   
   const FEValuesBase<dim>& fe_v_neighbor = info2.fe_values();
   const unsigned int dofs_per_cell_neighbor = fe_v_neighbor.dofs_per_cell;

   // Get cell number of two cells adjacent to this face
   //const unsigned int cell_no          = cell_number (dinfo1.cell);
   //const unsigned int neighbor_cell_no = cell_number (dinfo2.cell);
   /*
   double mu_shock_avg = 0.5 * 
      (mu_shock(cell_no)/dinfo1.cell->diameter()
       +
       mu_shock(neighbor_cell_no)/dinfo2.cell->diameter());
   */
   std::vector<Sacado::Fad::DFad<double> >
   independent_local_dof_values (dofs_per_cell),
   independent_neighbor_dof_values (dofs_per_cell_neighbor);
   
   const unsigned int n_independent_variables = dofs_per_cell + 
                                                dofs_per_cell_neighbor;
   
   for (unsigned int i = 0; i < dofs_per_cell; i++)
   {
      independent_local_dof_values[i] = primal_solution(dof_indices[i]);
      independent_local_dof_values[i].diff(i, n_independent_variables);
   }
   
   for (unsigned int i = 0; i < dofs_per_cell_neighbor; i++)
   {
      independent_neighbor_dof_values[i] = primal_solution(dof_indices_neighbor[i]);
      independent_neighbor_dof_values[i].diff(i+dofs_per_cell, n_independent_variables);
   }
   
   
   // Compute two states on the face
   Table<2,Sacado::Fad::DFad<double> >
   Wplus (n_q_points, EulerEquations<dim>::n_components),
   Wminus (n_q_points, EulerEquations<dim>::n_components);

   /* TODO:ADIFF
   Table<3,Sacado::Fad::DFad<double> >
   grad_Wplus  (n_q_points, EulerEquations<dim>::n_components, dim),
   grad_Wminus (n_q_points, EulerEquations<dim>::n_components, dim);
   */

   for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
         const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;
         Sacado::Fad::DFad<double> dof = independent_local_dof_values[i];
         Wplus[q][c] += dof * fe_v.shape_value_component(i, q, c);

         /* TODO:ADIFF
         for (unsigned int d = 0; d < dim; d++)
            grad_Wplus[q][c][d] += dof *
                                   fe_v.shape_grad_component(i, q, c)[d];
         */
      }
   
   // Wminus is Neighbouring cell value
   for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell_neighbor; ++i)
      {
         const unsigned int c = fe_v_neighbor.get_fe().system_to_component_index(i).first;
         Sacado::Fad::DFad<double> dof = independent_neighbor_dof_values[i];
                                          
         Wminus[q][c] += dof * fe_v_neighbor.shape_value_component(i, q, c);

         /* TODO:ADIFF
         for (unsigned int d = 0; d < dim; d++)
            grad_Wminus[q][c][d] += dof *
                                    fe_v_neighbor.shape_grad_component(i, q, c)[d];
         */
      }
   
   
   // Compute numerical flux at all quadrature points
   //typedef Sacado::Fad::DFad<double> NormalFlux[EulerEquations<dim>::n_components];
   //NormalFlux *normal_fluxes = new NormalFlux[n_q_points];
   std::vector<std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components>> 
                                                               normal_fluxes(n_q_points);

   for (unsigned int q=0; q<n_q_points; ++q)
      numerical_normal_flux(fe_v.normal_vector(q),
                            Wplus[q],
                            Wminus[q],
                            //cell_average[cell_no],
                            //cell_average[neighbor_cell_no],
                            normal_fluxes[q]);
   
   // Now assemble the face term
   for (unsigned int i=0; i<dofs_per_cell; ++i)
      if (fe_v.get_fe().has_support_on_face(i, face_no) == true) // comment for TODO:ADIFF
      {
         Sacado::Fad::DFad<double> F_i = 0;
         
         for (unsigned int point=0; point<n_q_points; ++point)
         {
            const unsigned int
            component_i = fe_v.get_fe().system_to_component_index(i).first;
            
            F_i += normal_fluxes[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);

            /* TODO:ADIFF
            F_i += 5.0 * mu_shock_avg *
                   (Wplus[point][component_i] - Wminus[point][component_i]) *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);

            for(unsigned int d=0; d<dim; ++d)
               F_i -= 0.5 *
                     (mu_shock(cell_no)         * grad_Wplus[point][component_i][d] +
                     mu_shock(neighbor_cell_no) * grad_Wminus[point][component_i][d]) *
                     fe_v.normal_vector(point)[d] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point)
                     +
                     0.5 *
                     mu_shock(cell_no) * 
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.normal_vector(point)[d] *
                     (Wplus[point][component_i] - Wminus[point][component_i]) *
                     fe_v.JxW(point);
                     */
         }
         
         for (unsigned int k=0; k<dofs_per_cell; ++k)
            local_matrix11 (i,k) += F_i.fastAccessDx(k);

         for (unsigned int k=0; k<dofs_per_cell_neighbor; ++k)
            local_matrix12(i,k) += F_i.fastAccessDx(dofs_per_cell+k);
         
         /* used to test if I can assemble the system_matrix in transposed order, but failed 
         for (unsigned int k=0; k<dofs_per_cell; ++k)
            local_matrix11 (k,i) += F_i.fastAccessDx(k);

         for (unsigned int k=0; k<dofs_per_cell_neighbor; ++k)
            local_matrix12(k,i) += F_i.fastAccessDx(dofs_per_cell+k);
         */
         //local_vector(i) -= F_i.val();
         //local_vector.print();     //TO SHOW THE LOCAL_VECTOR
         //std::cout<<"this is interior_fece_term of this face on cell: "<<cell_no<<std::endl;
      }
   
   // Contributions to neighbouring cell
   for (unsigned int i=0; i<dofs_per_cell_neighbor; ++i)
      if (fe_v_neighbor.get_fe().has_support_on_face(i, face_no_neighbor) == true) // comment for TODO:ADIFF
      {
         Sacado::Fad::DFad<double> F_i = 0;
         
         for (unsigned int point=0; point<n_q_points; ++point)
         {
            const unsigned int
            component_i = fe_v_neighbor.get_fe().system_to_component_index(i).first;
            
            F_i -= normal_fluxes[point][component_i] *
                   fe_v_neighbor.shape_value_component(i, point, component_i) *
                   fe_v_neighbor.JxW(point);

            /* TODO:ADIFF
            F_i -= 5.0 * mu_shock_avg *
                   (Wplus[point][component_i] - Wminus[point][component_i]) *
                   fe_v_neighbor.shape_value_component(i, point, component_i) *
                   fe_v_neighbor.JxW(point);

            for(unsigned int d=0; d<dim; ++d)
               F_i += 0.5 *
                     (mu_shock(cell_no)         * grad_Wplus[point][component_i][d] +
                     mu_shock(neighbor_cell_no) * grad_Wminus[point][component_i][d]) *
                     fe_v.normal_vector(point)[d] *
                     fe_v_neighbor.shape_value_component(i, point, component_i) *
                     fe_v_neighbor.JxW(point)
                     -
                     0.5 *
                     mu_shock(neighbor_cell_no) * 
                     fe_v_neighbor.shape_grad_component(i, point, component_i)[d] *
                     fe_v.normal_vector(point)[d] *
                     (Wplus[point][component_i] - Wminus[point][component_i]) *
                     fe_v_neighbor.JxW(point);
                     */
         }
         
         for (unsigned int k=0; k<dofs_per_cell_neighbor; ++k)
            local_matrix_neighbor22 (i,k) += F_i.fastAccessDx(dofs_per_cell+k);
         
         for (unsigned int k=0; k<dofs_per_cell; ++k)
            local_matrix_neighbor21 (i,k) += F_i.fastAccessDx(k);
         
         /* used to test if I can assemble the system_matrix in transposed order, but failed 
         for (unsigned int k=0; k<dofs_per_cell_neighbor; ++k)
            local_matrix_neighbor22 (k,i) += F_i.fastAccessDx(dofs_per_cell+k);
         
         for (unsigned int k=0; k<dofs_per_cell; ++k)
            local_matrix_neighbor21 (k,i) += F_i.fastAccessDx(k);
         */
         //local_vector_neighbor (i) -= F_i.val();
         //local_vector.print();     //TO SHOW THE LOCAL_VECTOR
         //std::cout<<"this is interior_fece_term of neighboring face on cell: "<<cell_no<<std::endl;
      }
   
   //delete[] normal_fluxes;
}

template class DualSolver<2>;

