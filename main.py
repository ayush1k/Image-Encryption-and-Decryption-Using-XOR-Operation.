import numpy as np
from PIL import Image
import random

def create_initial_population(size, genome_length):
    return [np.random.randint(0, 256, genome_length).astype(np.uint8) for _ in range(size)]

def fitness(key, target):
    return np.sum(key == target)

def selection(population, scores, k=3):
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k-1):
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

def crossover(p1, p2):
    c = random.randint(1, len(p1) - 2)
    child1 = np.concatenate([p1[:c], p2[c:]])
    child2 = np.concatenate([p2[:c], p1[c:]])
    return child1, child2


def mutation(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = np.uint8(random.randint(0, 255))
    return individual

def genetic_algorithm(target, pop_size, genome_length, n_generations):
    population = create_initial_population(pop_size, genome_length)
    best = None
    best_eval = 0
    for gen in range(n_generations):
        scores = [fitness(ind, target) for ind in population]
        for i in range(len(population)):
            if scores[i] > best_eval:
                best, best_eval = population[i], scores[i]
                print(f"New best score: {best_eval}")
                if best_eval == len(target):
                    return best
        selected = [selection(population, scores) for _ in range(len(population))]
        children = []
        for i in range(0, len(population), 2):
            if i + 1 < len(population):  # ensure there's a pair to crossover
                p1, p2 = selected[i], selected[i+1]
                child1, child2 = crossover(p1, p2)
                children.append(mutation(child1))
                children.append(mutation(child2))
        population = children
    return best


def encrypt_decrypt_image(image, key):
    image_data = np.array(image)
    flat_image = image_data.flatten()
    encrypted_decrypted = np.bitwise_xor(flat_image, key[:len(flat_image)]).reshape(image_data.shape)
    return Image.fromarray(encrypted_decrypted)

# Main Execution Flow
if __name__ == "__main__":
    # Load image
    input_image = Image.open('input_image.jpg').convert('L')  # Convert image to grayscale
    input_image_data = np.array(input_image)
    
    # Parameters
    POP_SIZE = 100
    GENOME_LENGTH = input_image_data.size
    N_GENERATIONS = 50
    
    # Generate a random target key (normally this would be your secret encryption key)
    target_key = np.random.randint(0, 256, GENOME_LENGTH).astype(np.uint8)

    # Run GA to find the encryption key
    found_key = genetic_algorithm(target_key, POP_SIZE, GENOME_LENGTH, N_GENERATIONS)
    
    # Encrypt and decrypt image
    encrypted_image = encrypt_decrypt_image(input_image, found_key)
    decrypted_image = encrypt_decrypt_image(encrypted_image, found_key)
    
    # Save or display results
    encrypted_image.save('encrypted_image.png')
    decrypted_image.save('decrypted_image.png')
    decrypted_image.show()
