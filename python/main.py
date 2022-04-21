#!/usr/bin/env python3

import pipeline_bhm

if __name__ == "__main__":

    pipeline_bhm.literature_convergence(perc_convergence=95.)
    # pipeline_bhm.patient_convergence(patient_number=1, perc_convergence=95., fixed_sd=10)
    # pipeline_bhm.patient_convergence(patient_number=1, perc_convergence=95., fixed_sd=5)
    # pipeline_bhm.patient_convergence(patient_number=1, perc_convergence=95., fixed_sd=1)
    pipeline_bhm.patient_convergence(patient_number=2, perc_convergence=95., fixed_sd=10)
    pipeline_bhm.patient_convergence(patient_number=10, perc_convergence=95., fixed_sd=10)
    pipeline_bhm.patient_convergence(patient_number=18, perc_convergence=95., fixed_sd=10)
    # pipeline_bhm.patient_convergence(patient_number=18, perc_convergence=95., fixed_sd=5)
    # pipeline_bhm.patient_convergence(patient_number=18, perc_convergence=95., fixed_sd=1)