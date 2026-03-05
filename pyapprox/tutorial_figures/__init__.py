# _figures — Tutorial plotting functions
#
# This package extracts matplotlib plotting code from tutorial .qmd files
# to keep tutorial cells clean and focused on computation / pedagogy.
#
# Two conventions:
#   Convention A (echo:false cells): Figure generators that accept PyApprox
#       objects and handle all compute + plot internally.
#   Convention B (echo:true cells): Pure plot functions that accept
#       pre-computed arrays and only handle plotting.
#
# All public functions are re-exported here for easy import:
#   from _figures import plot_whatever

from ._forward_uq import (  # noqa: F401
    plot_forward_uq_schematic,
    plot_linear_propagation,
    plot_nonlinear_propagation,
    plot_different_priors,
    plot_different_models,
    plot_summary_statistics,
    plot_indep_beta,
    plot_copula_beta,
    plot_two_priors,
    plot_beam_deflections,
    plot_samples_and_surface,
    plot_mc_variability_histogram,
    plot_mc_variability_comparison,
    plot_multi_qoi_scatter,
    plot_mean_mse_convergence,
    plot_mean_vs_var_convergence,
    plot_estimator_covariance,
    plot_diagonal_variances,
)

from ._sensitivity import (  # noqa: F401
    plot_variance_decomposition,
    plot_sobol_bar_chart,
    plot_scatter_dominant,
    plot_sobol_indices,
    plot_group_pie,
    plot_bin_1d,
    plot_bin_2d,
    plot_bin_vs_mc,
    plot_bootstrap,
)

from ._cv_acv import (  # noqa: F401
    plot_model_surfaces,
    plot_variance_reduction_vs_rho,
    plot_cvmc_histograms,
    plot_cv_verification,
    plot_unknown_mean_problem,
    plot_acv_variance_reduction_vs_r,
    plot_direct_vs_indirect,
    plot_acv_ceiling,
    plot_acv_two_model_verification,
    plot_allocation_matrix,
    plot_allocation_matrices,
    plot_variance_verification,
)

from ._hierarchy import (  # noqa: F401
    plot_level_variances,
    plot_mlmc_vs_mc,
    plot_variance_vs_cost,
    plot_mlmc_vs_opt,
    plot_sample_structure,
    plot_mfmc_variance_verify,
)

from ._multifidelity_advanced import (  # noqa: F401
    plot_mlblue_subsets,
    plot_mlblue_system,
    plot_mlblue_cov,
    plot_mlblue_ceiling,
    plot_mlblue_verify,
    plot_mlblue_m_sweep,
    plot_pacv_dags,
    plot_pacv_family_matrix,
    plot_pacv_enumeration,
    plot_pacv_ceiling,
    plot_pacv_cross_family,
    plot_moacv_vs_soacv,
    plot_mo_theory_vs_empirical,
    plot_bad_model,
    plot_correlation_heatmaps,
    plot_ensemble_nmodels,
    plot_pilot_tradeoff,
    plot_budget_verification,
    plot_budget_vs_statistic,
    plot_pilot_sensitivity,
    plot_est_cov_final,
)

from ._interpolation import (  # noqa: F401
    plot_lagrange_basis,
    plot_basis_comparison,
    plot_runge,
    plot_nested_cc,
    plot_interp_convergence,
    plot_tp_2d,
    plot_curse_of_dimensionality,
    plot_2d_convergence,
    plot_gibbs,
    plot_piecewise_basis,
    plot_pw_convergence,
    plot_discontinuous,
    plot_leja_growth,
    plot_node_distribution,
    plot_lagrange_on_leja,
    plot_leja_convergence,
    plot_two_point_quadrature,
    plot_leja_beta,
)

from ._quadrature import (  # noqa: F401
    plot_gauss_nodes,
    plot_quad_convergence_comparison,
    plot_gauss_hermite,
    plot_lobatto,
    plot_newton_cotes,
    plot_pw_quad_convergence,
    plot_smolyak_combo,
    plot_sg_points,
    plot_point_counts,
    plot_4d_convergence,
    plot_growth_rules,
    plot_additive_function,
    plot_index_set,
    plot_points_compare,
    plot_adaptive_vs_iso,
    plot_4d_adaptive,
)

from ._sparse_grids import (  # noqa: F401
    plot_model_hierarchy,
    plot_config_vars,
    plot_mf_indices_1d,
    plot_mf_animation,
    plot_mf_manual_convergence,
    plot_cost_weighted_animation,
    plot_mf_indices_2d,
    plot_mf_2d_accuracy,
    plot_sg_sobol,
    plot_sg_marginals,
    plot_sg_vs_mc,
    plot_two_bases,
    plot_projection_1d,
    plot_2d_tp,
    plot_smolyak_merge,
    plot_sg_vs_pce,
)

from ._gp import (  # noqa: F401
    plot_prior_posterior_samples,
    plot_kernel_comparison,
    plot_nlml_landscape,
    plot_length_scales,
    plot_gp_predictions,
    plot_calibration,
    plot_gp_convergence_n,
    plot_uncertainty_map,
    plot_gp_sobol,
    plot_sobol_distribution,
    plot_forrester_functions,
    plot_predictions_uncertainty,
    plot_kernel_matrix,
    plot_sample_designs,
    plot_gp_sampling_convergence,
)

from ._pce import (  # noqa: F401
    plot_pce_predictions,
    plot_pce_convergence,
    plot_truncation_error,
    plot_train_test_cv,
    plot_train_test_cv_2panel,
    plot_coefficient_decay,
    plot_omp_residual,
    plot_sparse_coefficients,
    plot_accuracy_vs_n,
    plot_pce_marginal_densities,
    plot_pce_vs_mc,
    plot_gp_marginal_density,
)

from ._function_train import (  # noqa: F401
    plot_ft_rank1_contours,
    plot_ft_highrank_contours,
    plot_core_diagram,
    plot_ft_3var,
    plot_ft_fitted,
    plot_ft_rank_comparison,
    plot_ft_sobol,
    plot_ft_decomposition,
    plot_ft_vs_mc,
)

from ._kle import (  # noqa: F401
    plot_cov_matrices,
    plot_eigenvalues_eigenfunctions,
    plot_sample_paths,
    plot_kle_truncation_error,
    plot_pointwise_variance,
    plot_analytical_validation,
    plot_mesh_kle_overview,
    plot_data_driven_eigenfunctions,
    plot_eigenvalue_convergence,
    plot_mesh_convergence,
    plot_field_reconstruction,
    plot_matern_kernels_paths,
    plot_spde_overview,
    plot_boundary_artefacts,
    plot_memory_scaling,
    plot_spde_parameters,
    plot_method_comparison,
)

from ._bayesian import (  # noqa: F401
    plot_forward_inverse_schematic,
    plot_point_estimate_problem,
    plot_likelihood,
    plot_bayes_update,
    plot_prior_likelihood_balance,
    plot_multiple_observations,
    plot_posterior_pushforward,
    plot_random_walk_concept,
    plot_mh_single_step,
    plot_trace_and_histogram,
    plot_2d_chain,
    plot_2d_marginals,
    plot_posterior_predictive,
    plot_proposal_width,
    plot_acceptance_rate,
    plot_burnin,
    plot_autocorrelation,
    plot_delayed_rejection,
    plot_adaptive_proposal,
    plot_dram_comparison,
)

from ._vi import (  # noqa: F401
    plot_two_strategies,
    plot_candidate_gallery,
    plot_optimization_snapshots,
    plot_vi_result,
    plot_vi_vs_mcmc,
    plot_bimodal,
    plot_family_comparison,
    plot_beta_vi,
    plot_overlap_intuition,
    plot_kl_landscape,
    plot_elbo_two_terms,
    plot_reparam_diagram,
    plot_reparam_visual,
    plot_elbo_convergence,
    plot_base_samples,
    plot_vi_2d,
    plot_amortization_concept,
    plot_training_recovery,
    plot_generalization,
)

from ._boed import (  # noqa: F401
    plot_kl_intuition,
    plot_eig_as_average,
    plot_eig_vs_nobs,
    plot_posterior_shrinkage,
    plot_boed_convergence_panels,
    plot_double_loop,
    plot_eig_landscape,
    plot_lv_trajectories,
    plot_lv_design,
    plot_points_comparison,
    plot_mc_vs_qmc,
    plot_design_variability,
    plot_design_convergence,
    plot_pushforward_intuition,
    plot_utility_as_average,
    plot_pushforward_shrinkage,
    plot_pred_mse_mc,
    plot_lv_pred_target,
    plot_pred_design_weights,
    plot_advec_diff_design,
)

from ._design import (  # noqa: F401
    plot_beam_setup,
    plot_two_posteriors,
    plot_response_surfaces,
    plot_combined_experiments,
    plot_eig_sweep,
    plot_eig_2sensor_heatmap,
    plot_duu_beam_schematic,
    plot_duu_comparison,
    plot_mdu_beam_geometry,
    plot_reference_solution,
    plot_uncertainty_sources,
)
