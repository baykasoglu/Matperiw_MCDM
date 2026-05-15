from MATPERIW_HOLACRATIC_corrected import HolacraticMatperiwMCDM, plot_final_ranking

model = HolacraticMatperiwMCDM(
    excel_path="matperiw_holacratic_brand_sustainability_input.xlsx",
    structure_sheet="Structure",
    alternatives_sheet="Alternatives",
    criterion_types_sheet="CriterionTypes",
    ahp_weight_derivation="max_eigen",
    cr_threshold=0.10,
    default_criterion_type=1,
)

output = model.run()

model.print_final_results()

model.export_results("matperiw_holacratic_brand_sustainability_results1.xlsx")

plot_final_ranking(output["final_result"]["ranking_table"])
