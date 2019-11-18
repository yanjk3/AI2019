from pomegranate import *
B = DiscreteDistribution({'T':0.001, 'F':0.999})
E = DiscreteDistribution({'T':0.002, 'F':0.998})
A = ConditionalProbabilityTable(
    [
        ['T', 'T', 'T', 0.95],
        ['T', 'T', 'F', 0.05],
        ['T', 'F', 'T', 0.94],
        ['T', 'F', 'F', 0.06],
        ['F', 'T', 'T', 0.29],
        ['F', 'T', 'F', 0.71],
        ['F', 'F', 'T', 0.001],
        ['F', 'F', 'F', 0.999]
    ], [B, E])
J = ConditionalProbabilityTable(
    [
        ['T', 'T', 0.90],
        ['T', 'F', 0.10],
        ['F', 'T', 0.05],
        ['F', 'F', 0.95],
    ], [A]
)
M = ConditionalProbabilityTable(
    [
        ['T', 'T', 0.70],
        ['T', 'F', 0.30],
        ['F', 'T', 0.01],
        ['F', 'F', 0.99],
    ], [A]
)
s1 = State(B, name='B')
s2 = State(E, name='E')
s3 = State(A, name='A')
s4 = State(J, name='J')
s5 = State(M, name='M')

model = BayesianNetwork('Problem1')

model.add_states(s1, s2, s3, s4, s5)
model.add_transition(s1, s3)
model.add_transition(s2, s3)
model.add_transition(s3, s4)
model.add_transition(s3, s5)
model.bake()

l = ['T', 'F']
ans1 = 0
ans2 = 0
for a in range(2):
    for b in range(2):
        for c in range(2):
            for d in range(2):
                ans1 += model.probability([l[a], l[b], 'T', l[c], l[d]])

for a in range(2):
    for b in range(2):
        for c in range(2):
            ans2 += model.probability([l[a], l[b], l[c], 'T', 'F'])

ans3 = model.predict_proba({'J':'T', 'M':'F'})[2].parameters[0]['T']
ans4 = model.predict_proba({'A':'T'})[0].parameters[0]['T']
ans5 = model.predict_proba({'J':'T', 'M':'F'})[0].parameters[0]['T']
ans6 = (1-ans5)*ans2/0.999
print('P(A): {:.8f}'.format(ans1))
print('P(J&&~M): {:.8f}'.format(ans2))
print('P(A|J&&~M): {:.8f}'.format(ans3))
print('P(B|A): {:.8f}'.format(ans4))
print('P(B|J&&~M): {:.8f}'.format(ans5))
print('P(J&&~M|~B): {:.8f}'.format(ans6))
