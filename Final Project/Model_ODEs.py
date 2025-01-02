# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Date : December 20, 2024
# Microbiome-Pathogen Interactions Drive Epidemiological Dynamics of ABR

# Final Project Model Implementation : Model_ODEs.py

# -------------------------------- MODEL ODE SYSTEMS ---------------------------------------

# Model 1 : Susceptible-Colonized Model 
def Susceptible_Colonized_ODE(t, U, Parameters):
    S, CR, CS_TNS, CS_ACQ, CR_TNS, CR_ACQ, CR_HGT = U
    beta, alpha, gamma, c, mu, fd, fC, fR, fw, a, rS, rR, theta_C, theta_M, epsilon, eta, phi, chi_e, chi_d, delta, omega = Parameters

    # Calculate compound parameter vlaues 
    alpha_R = alpha * (1 - a * (1 - rR))
    gamma_R = gamma * (1 + c)
    sigma_R = a * (1 - rR) * theta_C

    # ODE System
    dS_dt = (1 - (fC * fR)) * mu - (S * mu) - S * (beta * CR + alpha_R) + CR * (gamma_R + sigma_R)
    dCR_dt = (fC * fR) * mu - (CR * mu) + S * (beta * CR + alpha_R) - CR * (gamma_R + sigma_R)

    # Incidence 
    dCS_TNS_dt = 0
    dCS_ACQ_dt = 0
    dCR_TNS_dt = beta * CR * S
    dCR_ACQ_dt = alpha_R * S
    dCR_HGT_dt = 0

    return [dS_dt, dCR_dt, 
            dCS_TNS_dt, dCS_ACQ_dt, dCR_TNS_dt, dCR_ACQ_dt, dCR_HGT_dt]

# -------------------------------------------------------------------------------------------
# Model 2 : Strain Competition Model 
def Strain_Competition_ODE(t, U, Parameters):
    S, CS, CR, CS_TNS, CS_ACQ, CR_TNS, CR_ACQ, CR_HGT = U
    beta, alpha, gamma, c, mu, fd, fC, fR, fw, a, rS, rR, theta_C, theta_M, epsilon, eta, phi, chi_e, chi_d, delta, omega = Parameters

    # Calculate compound parameter values 
    alpha_S = alpha * (1 - a * (1- rS))
    alpha_R = alpha * (1 - a * (1- rR))
    gamma_S = gamma
    gamma_R = gamma * (1 + c)
    sigma_S = a * (1 - rS) * theta_C
    sigma_R = a * (1 - rR) * theta_C

    # ODE System
    dS_dt = (1 - fC) * mu - (S * mu) - S * (beta * (CS + CR) + alpha_S + alpha_R) + CS * (gamma_S + sigma_S) + CR * (gamma_R + sigma_R)
    dCS_dt = fC * (1 - fR) * mu - (CS * mu) + S * (beta * CS + alpha_S) - CS * (gamma_S + sigma_S)
    dCR_dt = fC * fR * mu - (CR * mu) + S * (beta * CR + alpha_R) - CR * (gamma_R + sigma_R)

    # Incidence
    dCS_TNS_dt = beta * CS * S
    dCS_ACQ_dt = alpha_S * S
    dCR_TNS_dt = beta * CR * S
    dCR_ACQ_dt = alpha_R * S
    dCR_HGT_dt = 0 

    return [dS_dt, dCS_dt, dCR_dt, 
            dCS_TNS_dt, dCS_ACQ_dt, dCR_TNS_dt, dCR_ACQ_dt, dCR_HGT_dt]

# -------------------------------------------------------------------------------------------
# Model 3 : Microbiome Competition Model 
def Microbiome_Competition_ODE(t, U, Parameters): 
    Se, Sd, CRe, CRd, CS_TNS, CS_ACQ, CR_TNS, CR_ACQ, CR_HGT = U
    beta, alpha, gamma, c, mu, fd, fC, fR, fw, a, rS, rR, theta_C, theta_M, epsilon, eta, phi, chi_e, chi_d, delta, omega = Parameters
    
    # Caculate compound parameter values 
    beta_epsilon = beta * (1 - epsilon)
    alpha_R = alpha * (1 - a * (1 - rR))
    alpha_R_phi = alpha_R * phi 
    gamma_R = gamma * (1 + c)
    gamma_R_eta = gamma_R * (1 - eta)
    sigma_R = a * (1 - rR) * theta_C
    sigma_M = a * theta_M
    delta = delta * (1 - a)

    # ODE System
    dSe_dt = (1 - (fC * fR)) * (1 - fd) * mu - (Se * mu) - Se * (beta_epsilon * (CRe + CRd) + alpha_R + sigma_M) + Sd * delta + CRe * (gamma_R + sigma_R)
    dSd_dt = (1 - (fC * fR)) * fd * mu - (Sd * mu) + Se * sigma_M - Sd * (beta *(CRe + CRd) + alpha_R_phi + delta) + CRd * (gamma_R_eta + sigma_R)
    dCRe_dt = fC * fR * (1 - fd) * mu - (CRe * mu) + Se * (beta_epsilon * (CRe + CRd) + alpha_R) - CRe * (gamma_R + sigma_M + sigma_R) + (CRd * delta)
    dCRd_dt = fC * fR * fd * mu - (CRd * mu) + Sd * (beta*(CRe + CRd) + alpha_R_phi) + CRe * sigma_M - CRd * (gamma_R_eta + delta + sigma_R)

    # Incidence 
    dCS_TNS_dt = 0
    dCS_ACQ_dt = 0
    dCR_TNS_dt = beta_epsilon * (CRe + CRd) * Se + beta*(CRe + CRd) * Sd
    dCR_ACQ_dt = alpha_R * (Se) + alpha_R_phi * (Sd)
    dCR_HGT_dt = 0 

    return [dSe_dt, dSd_dt, dCRe_dt, dCRd_dt, 
            dCS_TNS_dt, dCS_ACQ_dt, dCR_TNS_dt, dCR_ACQ_dt, dCR_HGT_dt]

# -------------------------------------------------------------------------------------------
# Model 4 : Two-Strain Microbiome Competition Model 
def TwoStrainBiome_Competition_ODE(t, U, Parameters): 
    Se, Sd, CSe, CSd, CRe, CRd, CS_TNS, CS_ACQ, CR_TNS, CR_ACQ, CR_HGT = U
    beta, alpha, gamma, c, mu, fd, fC, fR, fw, a, rS, rR, theta_C, theta_M, epsilon, eta, phi, chi_e, chi_d, delta, omega = Parameters
    
    # Calculate compound parameter values 
    beta_epsilon = beta * (1 - epsilon)
    alpha_S = alpha * (1 - a * (1 - rS))
    alpha_R = alpha * (1 - a * (1 - rR))
    alpha_S_phi = alpha_S * phi 
    alpha_R_phi = alpha_R * phi 
    gamma_S = gamma 
    gamma_R = gamma * (1 + c)
    gamma_S_eta = gamma_S * (1 - eta)
    gamma_R_eta = gamma_R * (1 - eta)
    sigma_S = a * (1 - rS) * theta_C
    sigma_R = a * (1 - rR) * theta_C
    sigma_M = a * theta_M
    delta = delta * (1 - a)

    # ODE System
    dSe_dt = (1- (fC)) * (1 - fd) * mu - (Se * mu) - Se * (beta_epsilon * (CSe + CSd + CRe + CRd) + alpha_S + alpha_R + sigma_M) + Sd * delta + CSe * (gamma_S + sigma_S) + CRe * (gamma_R + sigma_R)
    dSd_dt = (1 - (fC * fR)) * fd * mu - (Sd * mu) + Se * sigma_M - Sd * (beta * (CRe + CRd) + alpha_R_phi + delta) + CRd * (gamma_R_eta + sigma_R)
    dCSe_dt = (1 - (fC)) * fd * mu - (Sd * mu) + Se * sigma_M - Sd * (beta * (CSe + CSd + CRe + CRd) + alpha_S_phi + alpha_R_phi + delta) + CSd * (gamma_S_eta + sigma_S) + CRd * (gamma_R_eta + sigma_R)
    dCSd_dt = fC * (1 - fR) * fd * mu - (CSd * mu) + Sd * (beta * (CSe + CSd) + alpha_S_phi) + CSe * sigma_M - CSd * (gamma_S_eta + delta + sigma_S)
    dCRe_dt = fC * fR * (1 - fd) * mu - (CRe * mu) + Se * (beta_epsilon * (CRe + CRd) + alpha_R) - CRe * (gamma_R + sigma_M + sigma_R) + (CRd * delta)
    dCRd_dt = fC * fR * fd * mu - (CRd * mu) + Sd * (beta * (CRe + CRd) + alpha_R_phi) + CRe * sigma_M - CRd * (gamma_R_eta + delta + sigma_R)

    # Incidence 
    dCS_TNS_dt = beta_epsilon * (CSe + CSd) * Se + beta*(CSe + CSd) * Sd
    dCS_ACQ_dt = alpha_S * (Se) + alpha_S_phi * (Sd)
    dCR_TNS_dt = beta_epsilon * (CRe + CRd) * Se + beta * (CRe + CRd) * Sd
    dCR_ACQ_dt = alpha_R * (Se) + alpha_R_phi * (Sd)
    dCR_HGT_dt = 0 

    return [dSe_dt, dSd_dt, dCSe_dt, dCSd_dt, dCRe_dt, dCRd_dt, 
            dCS_TNS_dt, dCS_ACQ_dt, dCR_TNS_dt, dCR_ACQ_dt, dCR_HGT_dt]

# -------------------------------------------------------------------------------------------
# Model 5 : Two-Strain Microbiome Competition Model with HGT
def TwoStrainHGT_Competition_ODE(t, U, Parameters): 
    SeS, SeR, SdS, SdR, CSeS, CSeR, CSdS, CSdR, CReS, CReR, CRdS, CRdR, CS_TNS, CS_ACQ, CR_TNS, CR_ACQ, CR_HGT = U
    beta, alpha, gamma, c, mu, fd, fC, fR, fw, a, rS, rR, theta_C, theta_M, epsilon, eta, phi, chi_e, chi_d, delta, omega = Parameters
    
    # Compute compounding parameter values 
    beta_epsilon = beta * (1 - epsilon)
    alpha_S = alpha * (1 - a * (1 - rS))
    alpha_R = alpha * (1 - a * (1 - rR))
    alpha_S_phi = alpha_S * phi 
    alpha_R_phi = alpha_R * phi 
    gamma_S = gamma 
    gamma_R = gamma * (1 + c)
    gamma_S_eta = gamma_S * (1 - eta)
    gamma_R_eta = gamma_R * (1 - eta)
    sigma_R = a * (1 - rR) * theta_C
    sigma_S = a * (1 - rS) * theta_C
    sigma_M = a * theta_M
    delta = delta * (1 - a)

    # ODE System
    dSeS_dt = (1 - (fC)) * (1 - fd) * (1 - fw) * mu - (SeS * mu) - SeS * (beta_epsilon * (CSeS + CSeR + CSdS + CSdR + CReS + CReR + CRdS + CRdR) + alpha_S + alpha_R + sigma_M) + SdS * delta + CSeS * (gamma_S + sigma_S) + CReS * (gamma_R + sigma_R)
    dSeR_dt = (1 - (fC)) * (1 - fd) * fw * mu - (SeR * mu) - SeR * (beta_epsilon * (CSeS + CSeR + CSdS + CSdR + CReS + CReR + CRdS + CRdR) + alpha_S + alpha_R + sigma_M) + SdR * delta + CSeR * (gamma_S + sigma_S) + CReR * (gamma_R + sigma_R)
    dSdS_dt = (1 - (fC)) * fd * (1 - fw) * mu - (SdS * mu) + SeS * ((1 - omega) * sigma_M) - SdS * (beta * (CSeS + CSeR + CSdS + CSdR + CReS + CReR + CRdS + CRdR) + alpha_S_phi + alpha_R_phi + delta) + CSdS * (gamma_S_eta + sigma_S) + CRdS * (gamma_R_eta + sigma_R)
    dSdR_dt = (1 - (fC)) * fd * fw * mu - (SdR * mu) + SeS * (omega * sigma_M) + SeR * sigma_M - SdR * (beta * (CSeS + CSeR + CSdS + CSdR + CReS + CReR + CRdS + CRdR) + alpha_S_phi + alpha_R_phi + delta) + CSdR * (gamma_S_eta + sigma_S) + CRdR * (gamma_R_eta + sigma_R)
    dCSeS_dt = fC * (1 - fR) * (1 - fd) * (1 - fw) * mu - (CSeS * mu) + SeS * (beta_epsilon * (CSeS + CSeR + CSdS + CSdR) + alpha_S) - CSeS * (gamma_S + sigma_M + sigma_S) + CSdS * delta
    dCSeR_dt = fC * (1 - fR) * (1 - fd) * fw * mu  - (CSeR * mu) + SeR * (beta_epsilon * (CSeS + CSeR + CSdS + CSdR) + alpha_S) - CSeR * (gamma_S + sigma_M + sigma_S + chi_e) + CSdR * delta
    dCSdS_dt = fC * (1 - fR) * fd * (1 - fw) * mu - (CSdS * mu) + SdS * (beta * (CSeS + CSeR + CSdS + CSdR) + alpha_S_phi) + CSeS * ((1 - omega) * sigma_M) - CSdS * (gamma_S_eta + delta + sigma_S)
    dCSdR_dt = fC * (1 - fR) * fd * fw * mu - (CSdR * mu) + SdR * (beta * (CSeS + CSeR + CSdS + CSdR) + alpha_S_phi) + CSeS * (omega * sigma_M) + CSeR * sigma_M - CSdR * (gamma_S_eta + delta + sigma_S + chi_d)
    dCReS_dt = fC * fR * (1 - fd) * (1 - fw) * mu - (CReS * mu) + SeS * (beta_epsilon * (CReS + CReR + CRdS + CRdR) + alpha_R) - CReS * (gamma_R + sigma_M + sigma_R + chi_e) + (CRdS * delta)
    dCReR_dt = fC * fR * (1 - fd) * fw * mu - (CReR * mu) + SeR * (beta_epsilon * (CReS + CReR + CRdS + CRdR) + alpha_R) + CSeR * chi_e + CReS * chi_e - CReR * (gamma_R + sigma_M + sigma_R) + CRdR * delta
    dCRdS_dt = fC * fR * fd * (1 - fw) * mu - (CRdS * mu) + SdS * (beta * (CReS + CReR + CRdS + CRdR) + alpha_R_phi) + CReS * ((1 - omega) * sigma_M) - CRdS * (gamma_R_eta + delta + sigma_R + chi_d)
    dCRdR_dt = fC * fR * fd * fw * mu - (CRdR * mu) + SdR * (beta * (CReS + CReR + CRdS + CRdR) + alpha_R_phi) + CSdR * chi_d + CReS * (omega * sigma_M) + CReR * sigma_M + CRdS * chi_d - CRdR * (gamma_R_eta + delta + sigma_R)

    # Incidence
    dCS_TNS_dt = beta_epsilon * (CSeS + CSeR + CSdS + CSdR) * SeS + beta_epsilon * (CSeS + CSeR + CSdS + CSdR) * SeR + beta * (CSeS + CSeR + CSdS + CSdR) * SdS + beta * (CSeS + CSeR + CSdS + CSdR) * SdR
    dCS_ACQ_dt = alpha_S * (SeS + SeR) + alpha_S_phi * (SdS + SdR)
    dCR_TNS_dt = beta_epsilon * (CReS + CReR + CRdS + CRdR) * SeS + beta_epsilon * (CReS + CReR + CRdS + CRdR) * SeR + beta * (CReS + CReR + CRdS + CRdR) * SdS + beta * (CReS + CReR + CRdS + CRdR) * SdR
    dCR_ACQ_dt = alpha_R * (SeS + SeR)+ alpha_R_phi * (SdS + SdR)
    dCR_HGT_dt = (chi_e) * (CSeR) + (chi_d) * (CSdR)

    return [dSeS_dt, dSeR_dt, dSdS_dt, dSdR_dt, dCSeS_dt, dCSeR_dt, dCSdS_dt, dCSdR_dt, dCReS_dt, dCReR_dt, dCRdS_dt, dCRdR_dt,
            dCS_TNS_dt, dCS_ACQ_dt, dCR_TNS_dt, dCR_ACQ_dt, dCR_HGT_dt]