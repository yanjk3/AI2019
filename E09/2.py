from pomegranate import *

PatientAge = DiscreteDistribution({'0-30':0.1, '31-65':0.3, '65+':0.6})
CTScanResult = DiscreteDistribution({'Ischemic Stroke':0.7, 'Hemmorraghic Stroke':0.3})
MRIScanResult = DiscreteDistribution({'Ischemic Stroke':0.7, 'Hemmorraghic Stroke':0.3})
Anticoagulants = DiscreteDistribution({'Used':0.5, 'Not used':0.5})

StrokeType = ConditionalProbabilityTable(
    [
        ['Ischemic Stroke', 'Ischemic Stroke', 'Ischemic Stroke', 0.8],
        ['Ischemic Stroke', 'Hemmorraghic Stroke', 'Ischemic Stroke', 0.5],
        ['Hemmorraghic Stroke', 'Ischemic Stroke', 'Ischemic Stroke', 0.5],
        ['Hemmorraghic Stroke', 'Hemmorraghic Stroke', 'Ischemic Stroke', 0.0],

        ['Ischemic Stroke', 'Ischemic Stroke', 'Hemmorraghic Stroke', 0.0],
        ['Ischemic Stroke', 'Hemmorraghic Stroke', 'Hemmorraghic Stroke', 0.4],
        ['Hemmorraghic Stroke', 'Ischemic Stroke', 'Hemmorraghic Stroke', 0.4],
        ['Hemmorraghic Stroke', 'Hemmorraghic Stroke', 'Hemmorraghic Stroke', 0.9],

        ['Ischemic Stroke', 'Ischemic Stroke', 'Stroke Mimic', 0.2],
        ['Ischemic Stroke', 'Hemmorraghic Stroke', 'Stroke Mimic', 0.1],
        ['Hemmorraghic Stroke', 'Ischemic Stroke', 'Stroke Mimic', 0.1],
        ['Hemmorraghic Stroke', 'Hemmorraghic Stroke', 'Stroke Mimic', 0.1]
    ], [ CTScanResult , MRIScanResult])

Mortality = ConditionalProbabilityTable(
    [
        ['Ischemic Stroke', 'Used', 'False', 0.28],
        ['Hemmorraghic Stroke', 'Used', 'False', 0.99],
        ['Stroke Mimic', 'Used', 'False', 0.1],

        ['Ischemic Stroke', 'Not used', 'False', 0.56],
        ['Hemmorraghic Stroke', 'Not used', 'False', 0.58],
        ['Stroke Mimic', 'Not used', 'False', 0.05],

        ['Ischemic Stroke', 'Used', 'True', 0.72],
        ['Hemmorraghic Stroke', 'Used', 'True', 0.01],
        ['Stroke Mimic', 'Used', 'True', 0.9],

        ['Ischemic Stroke', 'Not used', 'True', 0.44],
        ['Hemmorraghic Stroke', 'Not used', 'True', 0.42],
        ['Stroke Mimic', 'Not used', 'True', 0.95]
    ], [ StrokeType, Anticoagulants])

Disability = ConditionalProbabilityTable(
    [
        ['Ischemic Stroke', '0-30', 'Negligible', 0.80],
        ['Hemmorraghic Stroke', '0-30', 'Negligible', 0.70],
        ['Stroke Mimic', '0-30', 'Negligible', 0.90],
        ['Ischemic Stroke', '31-65', 'Negligible', 0.60],
        ['Hemmorraghic Stroke', '31-65', 'Negligible', 0.50],
        ['Stroke Mimic', '31-65', 'Negligible', 0.40],
        ['Ischemic Stroke', '65+', 'Negligible', 0.30],
        ['Hemmorraghic Stroke', '65+', 'Negligible', 0.20],
        ['Stroke Mimic', '65+', 'Negligible', 0.10],

        ['Ischemic Stroke', '0-30', 'Moderate', 0.10],
        ['Hemmorraghic Stroke', '0-30', 'Moderate', 0.20],
        ['Stroke Mimic', '0-30', 'Moderate', 0.05],
        ['Ischemic Stroke', '31-65', 'Moderate', 0.30],
        ['Hemmorraghic Stroke', '31-65', 'Moderate', 0.40],
        ['Stroke Mimic', '31-65', 'Moderate', 0.30],
        ['Ischemic Stroke', '65+', 'Moderate', 0.40],
        ['Hemmorraghic Stroke', '65+', 'Moderate', 0.20],
        ['Stroke Mimic', '65+', 'Moderate', 0.10],

        ['Ischemic Stroke', '0-30', 'Severe', 0.10],
        ['Hemmorraghic Stroke', '0-30', 'Severe', 0.10],
        ['Stroke Mimic', '0-30', 'Severe', 0.05],
        ['Ischemic Stroke', '31-65', 'Severe', 0.10],
        ['Hemmorraghic Stroke', '31-65', 'Severe', 0.10],
        ['Stroke Mimic', '31-65', 'Severe', 0.30],
        ['Ischemic Stroke', '65+', 'Severe', 0.30],
        ['Hemmorraghic Stroke', '65+', 'Severe', 0.60],
        ['Stroke Mimic', '65+', 'Severe', 0.80]
    ], [ StrokeType, PatientAge])

s1 = State(PatientAge, name='P')
s2 = State(CTScanResult, name='C')
s3 = State(MRIScanResult, name='MRI')
s4 = State(Anticoagulants, name='A')
s5 = State(StrokeType, name='S')
s6 = State(Mortality, name='M')
s7 = State(Disability, name='D')

model = BayesianNetwork('Problem2')
model.add_states(s1, s2, s3, s4, s5, s6, s7)
model.add_transition(s2, s5)
model.add_transition(s3, s5)
model.add_transition(s5, s6)
model.add_transition(s4, s6)
model.add_transition(s5, s7)
model.add_transition(s1, s7)
model.bake()

ans1 = model.predict_proba({'P':'31-65', 'C':'Ischemic Stroke'})[5].parameters[0]['True']
ans2 = model.predict_proba({'P':'65+', 'MRI':'Hemmorraghic Stroke'})[6].parameters[0]['Moderate']
ans3 = model.predict_proba({'P':'65+', 'C':'Hemmorraghic Stroke','MRI':'Ischemic Stroke'})[4].parameters[0]['Stroke Mimic']
ans4 = model.predict_proba({'P':'0-30'})[3].parameters[0]['Not used']
print ('P1: {:.8f}'.format(ans1))
print ('P2: {:.8f}'.format(ans2))
print ('P3: {:.8f}'.format(ans3))
print ('P4: {:.8f}'.format(ans4))
