# Dash SEIR Model

The SEIR model and code is based on Henri Froese's trilogy "Infectious Disease Modelling: Understanding the models that are used to model Coronavirus"
See here: https://towardsdatascience.com/infectious-disease-modelling-part-i-understanding-sir-28d60e29fdfc

I have ended up NOT introducing the planned modification for critical community size: any compartment having less than one person in, is truncated to 0, following "DETERMINISTIC CRITICAL COMMUNITY SIZE FOR THE SIR SYSTEM AND VIRAL STRAIN SELECTION" by MARCILIO FERREIRA DOS SANTOS CESAR CASTILHO
https://www.biorxiv.org/content/10.1101/2020.05.08.084673v1.full

I have changed the code to have all parameters randomly distributed (Monte Carlo) but CONSTANT as function of time, disabled excess death due to exceeding number of "Beds".

The peak value of the Critical compartment is reported using a Monte Carlo approach, with uniform distribution of the parameters as set at the beginning.

Relying on the guidance and extensive help of Steve Puts from World Infectious Disease Monitoring Organization (WoIDMo).
Ornit Hadar, June 23rd 2020
