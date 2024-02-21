from pyneurgen.grammatical_evolution import GrammaticalEvolution
from pyneurgen.fitness import FitnessElites, FitnessTournament
from pyneurgen.fitness import ReplacementTournament, MAX, MIN, CENTER


with open('/workspaces/pyneurgen/2/satelite_data/ImageExpertReduced.txt', 'r') as expert_file:
    ground_truth = expert_file.read()
ground_truth = str([int(float(x)) for x in ground_truth.split()])
print(ground_truth)
with open('/workspaces/pyneurgen/2/satelite_data/ImageRawReduced.txt', 'r') as data_file:
    satelite_images = data_file.read()
satelite_images = str([[float(x) for x in line.split()] for line in satelite_images.splitlines()])
print(satelite_images)


bnf =   """
<expr>              ::= <expr> <biop> <expr> | <uop> <expr> | <real> |
                        math.log(abs(<expr>)) | <pow> | math.sin(<expr> )| (<expr>)
<biop>              ::= + | - | * | /
<uop>               ::= + | -
<pow>               ::= pow(<expr>, <real>)
<plus>              ::= +
<minus>             ::= -
<real>              ::= <int-const>.<int-const>
<int-const>         ::= <int-const> | 1 | 2 | 3 | 4 | 5 | 6 |
                        7 | 8 | 9 | 0
<r11>               ::= <expr>
<r12>               ::= <expr>
<r13>               ::= <expr>
<r21>               ::= <expr>
<r22>               ::= <expr>
<r23>               ::= <expr>
<r31>               ::= <expr>
<r32>               ::= <expr>
<r33>               ::= <expr>
<S>                 ::=
import numpy as np
import math

rules = np.array([[<r11>, <r12>, <r13>], [<r21>, <r22>, <r23>], [<r31>, <r32>, <r33>]])

ground_truth = np.array({gt})

satelite_images = np.array({si})

signal = np.matmul(rules, satelite_images)

predictions = signal.argmax(axis=0) + 1

fitness = (predictions == ground_truth).sum()

self.set_bnf_variable('<fitness>', fitness)
        """.format(gt=ground_truth, si=satelite_images)

# print bnf

ges = GrammaticalEvolution()

ges.set_bnf(bnf)
ges.set_genotype_length(start_gene_length=20,
                        max_gene_length=500)
ges.set_population_size(50)
ges.set_wrap(True)

ges.set_max_generations(10)
ges.set_fitness_type(MAX, 9350.)

ges.set_max_program_length(500)
ges.set_timeouts(30, 120)
ges.set_fitness_fail(0)

ges.set_fitness_selections(
    FitnessElites(ges.fitness_list, .05),
    FitnessTournament(ges.fitness_list, tournament_size=2))
ges.set_max_fitness_rate(.5)

ges.set_mutation_rate(.025)
ges.set_crossover_rate(.2)
ges.set_children_per_crossover(2)
ges.set_mutation_type('m')
ges.set_max_fitness_rate(.25)

ges.set_replacement_selections(
        ReplacementTournament(ges.fitness_list, tournament_size=3))

ges.set_maintain_history(True)

ges.create_genotypes()
print ges.run()
