import numpy as np
from PIL import Image, ImageTk
import random
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

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

def plot_histograms(original_image, encrypted_image, decrypted_image):
    original_data = np.array(original_image).flatten()
    encrypted_data = np.array(encrypted_image).flatten()
    decrypted_data = np.array(decrypted_image).flatten()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.hist(original_data, bins=256, range=(0, 256), color='blue', alpha=0.7)
    ax1.set_title('Original Image Histogram')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(encrypted_data, bins=256, range=(0, 256), color='blue', alpha=0.7)
    ax2.set_title('Encrypted Image Histogram')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    
    ax3.hist(decrypted_data, bins=256, range=(0, 256), color='blue', alpha=0.7)
    ax3.set_title('Decrypted Image Histogram')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Frequency')
    
    fig.tight_layout()
    
    plt.show()

def plot_correlations(original_image, encrypted_image):
    original_data = np.array(original_image)
    encrypted_data = np.array(encrypted_image)
    
    original_flat = original_data.flatten()
    encrypted_flat = encrypted_data.flatten()
    
    original_pairs = np.vstack((original_flat[:-1], original_flat[1:])).T
    encrypted_pairs = np.vstack((encrypted_flat[:-1], encrypted_flat[1:])).T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.scatter(original_pairs[:, 0], original_pairs[:, 1], color='blue', alpha=0.5, s=1)
    ax1.set_title('Original Image Correlation')
    ax1.set_xlabel('Pixel Value (i)')
    ax1.set_ylabel('Pixel Value (i+1)')
    
    ax2.scatter(encrypted_pairs[:, 0], encrypted_pairs[:, 1], color='blue', alpha=0.5, s=1)
    ax2.set_title('Encrypted Image Correlation')
    ax2.set_xlabel('Pixel Value (i)')
    ax2.set_ylabel('Pixel Value (i+1)')
    
    fig.tight_layout()
    
    plt.show()

def calculate_correlation_coefficients(image):
    data = np.array(image)
    h, w = data.shape
    horizontal_pairs = np.vstack((data[:, :-1].flatten(), data[:, 1:].flatten()))
    vertical_pairs = np.vstack((data[:-1, :].flatten(), data[1:, :].flatten()))
    diagonal_pairs1 = np.vstack((data[:-1, :-1].flatten(), data[1:, 1:].flatten()))
    diagonal_pairs2 = np.vstack((data[:-1, 1:].flatten(), data[1:, :-1].flatten()))
    
    def correlation_coeff(pairs):
        return np.corrcoef(pairs)[0, 1]
    
    return (correlation_coeff(horizontal_pairs),
            correlation_coeff(vertical_pairs),
            correlation_coeff(diagonal_pairs1),
            correlation_coeff(diagonal_pairs2))

class ImageEncryptorDecryptorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Encryptor/Decryptor")
        self.root.geometry("800x600")
        
        self.input_image = None
        self.encrypted_image = None
        self.decrypted_image = None
        self.found_key = None

        self.frame1 = tk.Frame(root)
        self.frame2 = tk.Frame(root)
        self.frame3 = tk.Frame(root)

        self.create_frame1()
        self.create_frame2()
        self.create_frame3()

        self.show_frame(self.frame1)

    def show_frame(self, frame):
        frame.tkraise()

    def create_frame1(self):
        self.frame1.grid(row=0, column=0, sticky='nsew')
        self.frame1.grid_rowconfigure(0, weight=1)
        self.frame1.grid_columnconfigure(0, weight=1)
        upload_button = tk.Button(self.frame1, text="Upload Image", command=self.upload_image)
        upload_button.grid(row=0, column=0, padx=20, pady=20, sticky="n")

    def create_frame2(self):
        self.frame2.grid(row=0, column=0, sticky='nsew')
        self.frame2.grid_rowconfigure(0, weight=1)
        self.frame2.grid_columnconfigure(0, weight=1)
        encrypt_button = tk.Button(self.frame2, text="Encrypt", command=self.encrypt_image)
        encrypt_button.grid(row=0, column=0, padx=20, pady=20, sticky="n")

    def create_frame3(self):
        self.frame3.grid(row=0, column=0, sticky='nsew')
        self.frame3.grid_rowconfigure(0, weight=1)
        self.frame3.grid_rowconfigure(1, weight=1)
        self.frame3.grid_rowconfigure(2, weight=1)
        self.frame3.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self.frame3, width=300, height=300)
        self.canvas.grid(row=0, column=0, pady=10)
        self.key_label = tk.Label(self.frame3, text="Key: ")
        self.key_label.grid(row=1, column=0, pady=10)
        
        decrypt_button = tk.Button(self.frame3, text="Decrypt Image", command=self.decrypt_image)
        decrypt_button.grid(row=2, column=0, pady=10)
        back_button = tk.Button(self.frame3, text="Back", command=lambda: self.show_frame(self.frame2))
        back_button.grid(row=3, column=0, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.input_image = Image.open(file_path).convert('L')
            self.show_frame(self.frame2)

    def encrypt_image(self):
        if self.input_image:
            input_image_data = np.array(self.input_image)
            pop_size = 100
            genome_length = input_image_data.size
            n_generations = 50
            target_key = np.random.randint(0, 256, genome_length).astype(np.uint8)
            self.found_key = genetic_algorithm(target_key, pop_size, genome_length, n_generations)
            self.encrypted_image = encrypt_decrypt_image(self.input_image, self.found_key)
            
            self.key_label.config(text=f"Key: {self.found_key[:10]}...")  # Show a part of the key
            self.show_encrypted_image()
            self.show_frame(self.frame3)

    def show_encrypted_image(self):
        encrypted_image_resized = self.encrypted_image.resize((150, 150))
        image = ImageTk.PhotoImage(encrypted_image_resized)
        self.canvas.create_image(150, 150, image=image)
        self.canvas.image = image

    def decrypt_image(self):
        if self.encrypted_image and self.found_key is not None:
            self.decrypted_image = encrypt_decrypt_image(self.encrypted_image, self.found_key)
            self.show_decrypted_image()

    def show_decrypted_image(self):
        decrypt_window = tk.Toplevel(self.root)
        decrypt_window.title("Decrypted Image")
        decrypt_window.geometry("800x600")
        
        decrypted_image_resized = self.decrypted_image.resize((150, 150))
        image = ImageTk.PhotoImage(decrypted_image_resized)
        canvas = tk.Canvas(decrypt_window, width=300, height=300)
        canvas.create_image(150, 150, image=image)
        canvas.image = image
        canvas.pack(pady=10)

        fsim_value = ssim(np.array(self.input_image), np.array(self.decrypted_image))
        eps = 1e-10
        psnr_value = psnr(np.array(self.input_image), np.array(self.decrypted_image), data_range=255)
        if psnr_value == float('inf'):
            psnr_value = 10 * np.log10((255 ** 2) / (eps))
        
        fsim_label = tk.Label(decrypt_window, text=f"FSIM: {fsim_value:.4f}")
        fsim_label.pack(pady=10)
        psnr_label = tk.Label(decrypt_window, text=f"PSNR: {psnr_value:.4f} dB")
        psnr_label.pack(pady=10)

        self.plot_histograms_and_correlations()

    def plot_histograms_and_correlations(self):
        if self.input_image and self.encrypted_image and self.decrypted_image:
            hist_window = tk.Toplevel(self.root)
            hist_window.title("Histograms and Correlations")
            hist_window.geometry("1200x800")
            
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            
            original_data = np.array(self.input_image).flatten()
            encrypted_data = np.array(self.encrypted_image).flatten()
            decrypted_data = np.array(self.decrypted_image).flatten()
            
            axs[0, 0].hist(original_data, bins=256, range=(0, 256), color='blue', alpha=0.7)
            axs[0, 0].set_title('Original Image Histogram')
            axs[0, 0].set_xlabel('Pixel Value')
            axs[0, 0].set_ylabel('Frequency')
            
            axs[0, 1].hist(encrypted_data, bins=256, range=(0, 256), color='blue', alpha=0.7)
            axs[0, 1].set_title('Encrypted Image Histogram')
            axs[0, 1].set_xlabel('Pixel Value')
            axs[0, 1].set_ylabel('Frequency')
            
            axs[0, 2].hist(decrypted_data, bins=256, range=(0, 256), color='blue', alpha=0.7)
            axs[0, 2].set_title('Decrypted Image Histogram')
            axs[0, 2].set_xlabel('Pixel Value')
            axs[0, 2].set_ylabel('Frequency')
            
            original_pairs = np.vstack((original_data[:-1], original_data[1:])).T
            encrypted_pairs = np.vstack((encrypted_data[:-1], encrypted_data[1:])).T
            decrypted_pairs = np.vstack((decrypted_data[:-1], decrypted_data[1:])).T
            
            axs[1, 0].scatter(original_pairs[:, 0], original_pairs[:, 1], color='blue', alpha=0.5, s=1)
            axs[1, 0].set_title('Original Image Correlation')
            axs[1, 0].set_xlabel('Pixel Value (i)')
            axs[1, 0].set_ylabel('Pixel Value (i+1)')
            
            axs[1, 1].scatter(encrypted_pairs[:, 0], encrypted_pairs[:, 1], color='blue', alpha=0.5, s=1)
            axs[1, 1].set_title('Encrypted Image Correlation')
            axs[1, 1].set_xlabel('Pixel Value (i)')
            axs[1, 1].set_ylabel('Pixel Value (i+1)')
            
            axs[1, 2].scatter(decrypted_pairs[:, 0], decrypted_pairs[:, 1], color='blue', alpha=0.5, s=1)
            axs[1, 2].set_title('Decrypted Image Correlation')
            axs[1, 2].set_xlabel('Pixel Value (i)')
            axs[1, 2].set_ylabel('Pixel Value (i+1)')
            
            fig.tight_layout()
            
            horizontal_corr_original, vertical_corr_original, diagonal1_corr_original, diagonal2_corr_original = calculate_correlation_coefficients(self.input_image)
            horizontal_corr_encrypted, vertical_corr_encrypted, diagonal1_corr_encrypted, diagonal2_corr_encrypted = calculate_correlation_coefficients(self.encrypted_image)
            horizontal_corr_decrypted, vertical_corr_decrypted, diagonal1_corr_decrypted, diagonal2_corr_decrypted = calculate_correlation_coefficients(self.decrypted_image)
            
            corr_table = tk.Frame(hist_window)
            corr_table.pack(pady=10)
            
            headers = ["", "Horizontal", "Vertical", "Diagonal 1", "Diagonal 2"]
            data = [
                ["Original Image", horizontal_corr_original, vertical_corr_original, diagonal1_corr_original, diagonal2_corr_original],
                ["Encrypted Image", horizontal_corr_encrypted, vertical_corr_encrypted, diagonal1_corr_encrypted, diagonal2_corr_encrypted],
                ["Decrypted Image", horizontal_corr_decrypted, vertical_corr_decrypted, diagonal1_corr_decrypted, diagonal2_corr_decrypted],
            ]
            
            for i, header in enumerate(headers):
                label = tk.Label(corr_table, text=header, font=('Helvetica', 12, 'bold'))
                label.grid(row=0, column=i, padx=5, pady=5)
                
            for row_index, row in enumerate(data):
                for col_index, item in enumerate(row):
                    label = tk.Label(corr_table, text=f"{item:.4f}" if isinstance(item, float) else item, font=('Helvetica', 12))
                    label.grid(row=row_index+1, column=col_index, padx=5, pady=5)
            
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.draw()
            canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEncryptorDecryptorApp(root)
    root.mainloop()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""
code with psnr and fsim score along with correlation table, histogram plot and correlation plot
need to remove the fsim and replace it with ssim
"""