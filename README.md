# This is my first PyCharm / Git script, and I've got introduced to SIR model two and a half weeks ago... so pls excuse my rookie's mistakes.
# The SEIR model and code is based on Henri Froese's trilogy "Infectious Disease Modelling: Understanding the models that are used to model Coronavirus"
# See here: https://towardsdatascience.com/infectious-disease-modelling-part-i-understanding-sir-28d60e29fdfc
# I have ended up NOT introducing the planned modification for critical community size: any compartment having less than one person in, is truncated to 0, following
# "DETERMINISTIC CRITICAL COMMUNITY SIZE FOR THE SIR SYSTEM AND VIRAL STRAIN SELECTION" by MARCILIO FERREIRA DOS SANTOS CESAR CASTILHO
# https://www.biorxiv.org/content/10.1101/2020.05.08.084673v1.full
# I have changed the code to have all parameters being randomly distributed (Monte Carlo) but CONSTANT as function of time,
# disabled exsess death due to exceeding number of "Beds".
# I report the peak value of the Critical compartment using Monte Carlo approach, with uniform distribution of the parameters
# as set at the beginning.
# Relying on the guidance and extensive help of Steve Putz from World Infectious Disease Monitoring Foundation (WoIDMo)
# Ornit Hadar, June 23rd 2020