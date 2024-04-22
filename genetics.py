import random


class Genetics:
    def __init__(self, genes, chromosome):
        self.genes = genes
        self.chromosome = chromosome

    @classmethod
    def get_random_gene(cls, genes):
        return random.choice(genes)

    @classmethod
    def create_genome(cls, genome, target):
        return [Genetics.get_random_gene(genome) for _ in range(len(target))]

    def crossover(self, parent_2):
        child_chromosomes = []
        border_prob = 0.45

        for gp1, gp2 in zip(self.chromosome, parent_2.chromosome):
            prob = random.random()

            if prob < border_prob:
                child_chromosomes.append(gp1)
            elif border_prob <= prob < 2 * border_prob:
                child_chromosomes.append(gp2)
            else:
                # child_chromosomes.append(random.choice(self.genes))
                child_chromosomes.append(self.get_random_gene(self.genes))

        return Genetics(self.genes, child_chromosomes)

    def get_fitness(self, target):
        fitness = 0

        for gs, gt in zip(self.chromosome, target):
            if gs != gt:
                fitness += 1
        return fitness
