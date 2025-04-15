import numpy as np
from PIL import Image, ImageTk
import random
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.metrics import peak_signal_noise_ratio as psnr

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
    
    return fig

def plot_correlations(original_image, encrypted_image, decrypted_image):
    original_data = np.array(original_image)
    encrypted_data = np.array(encrypted_image)
    decrypted_data = np.array(decrypted_image)
    
    original_flat = original_data.flatten()
    encrypted_flat = encrypted_data.flatten()
    decrypted_flat = decrypted_data.flatten()
    
    original_pairs = np.vstack((original_flat[:-1], original_flat[1:])).T
    encrypted_pairs = np.vstack((encrypted_flat[:-1], encrypted_flat[1:])).T
    decrypted_pairs = np.vstack((decrypted_flat[:-1], decrypted_flat[1:])).T
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.scatter(original_pairs[:, 0], original_pairs[:, 1], color='blue', alpha=0.5, s=1)
    ax1.set_title('Original Image Correlation')
    ax1.set_xlabel('Pixel Value (i)')
    ax1.set_ylabel('Pixel Value (i+1)')
    
    ax2.scatter(encrypted_pairs[:, 0], encrypted_pairs[:, 1], color='blue', alpha=0.5, s=1)
    ax2.set_title('Encrypted Image Correlation')
    ax2.set_xlabel('Pixel Value (i)')
    ax2.set_ylabel('Pixel Value (i+1)')
    
    ax3.scatter(decrypted_pairs[:, 0], decrypted_pairs[:, 1], color='blue', alpha=0.5, s=1)
    ax3.set_title('Decrypted Image Correlation')
    ax3.set_xlabel('Pixel Value (i)')
    ax3.set_ylabel('Pixel Value (i+1)')
    
    fig.tight_layout()
    
    return fig

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
            self.target_key = np.random.randint(0, 256, self.input_image.size[0] * self.input_image.size[1], dtype=np.uint8)
            pop_size = 100
            n_generations = 200
            self.found_key = genetic_algorithm(self.target_key, pop_size, len(self.target_key), n_generations)
            self.encrypted_image = encrypt_decrypt_image(self.input_image, self.found_key)
            self.key_label.config(text=f"Key: {self.found_key}")
            self.show_frame(self.frame3)
            
            self.display_image(self.encrypted_image, self.canvas)
            
            fig_hist = plot_histograms(self.input_image, self.encrypted_image, self.input_image)
            fig_corr = plot_correlations(self.input_image, self.encrypted_image, self.input_image)
            self.show_plots(fig_hist, fig_corr)

    def decrypt_image(self):
        if self.encrypted_image and self.found_key is not None:
            self.decrypted_image = encrypt_decrypt_image(self.encrypted_image, self.found_key)
            decrypted_image_canvas = tk.Canvas(self.frame3, width=300, height=300)
            decrypted_image_canvas.grid(row=0, column=1, pady=10)
            self.display_image(self.decrypted_image, decrypted_image_canvas)
            
            fig_hist = plot_histograms(self.input_image, self.encrypted_image, self.decrypted_image)
            fig_corr = plot_correlations(self.input_image, self.encrypted_image, self.decrypted_image)
            self.show_plots(fig_hist, fig_corr)
            
            psnr_value = psnr(np.array(self.input_image), np.array(self.decrypted_image))
            messagebox.showinfo("PSNR", f"PSNR value: {psnr_value:.2f} dB")

            corr_coeffs = calculate_correlation_coefficients(self.decrypted_image)
            self.show_correlation_coefficients(corr_coeffs)

    def display_image(self, image, canvas):
        img = image.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(150, 150, image=img_tk)
        canvas.image = img_tk

    def show_plots(self, fig_hist, fig_corr):
        plot_window = Toplevel(self.root)
        plot_window.title("Histograms and Correlations")
        
        canvas_hist = FigureCanvasTkAgg(fig_hist, master=plot_window)
        canvas_hist.draw()
        canvas_hist.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        canvas_corr = FigureCanvasTkAgg(fig_corr, master=plot_window)
        canvas_corr.draw()
        canvas_corr.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def show_correlation_coefficients(self, coeffs):
        corr_window = Toplevel(self.root)
        corr_window.title("Correlation Coefficients")
        
        labels = ['Horizontal', 'Vertical', 'Diagonal (\\)', 'Diagonal (/)']
        for i, (label, coeff) in enumerate(zip(labels, coeffs)):
            tk.Label(corr_window, text=f"{label}: {coeff:.4f}").grid(row=i, column=0, padx=10, pady=5)

root = tk.Tk()
app = ImageEncryptorDecryptorApp(root)
root.mainloop()
















