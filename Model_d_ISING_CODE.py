# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: modele_ising_econophysique.py
# execution: true

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# Fonctions utilitaires
def initialiser_reseau(n, K):
    """
    Initialise aléatoirement un réseau de spins pour K actions
    n: taille du réseau (n x n)
    K: nombre d'actions
    Retourne: S de taille n x n x K avec des valeurs +1 ou -1
    """
    return 2 * (np.random.rand(n, n, K) > 0.5).astype(int) - 1

def calculer_magnetisation(S):
    """
    Calcule la magnétisation moyenne pour chaque action
    S: réseau de spins (n x n x K)
    Retourne: vecteur M de taille K
    """
    n, _, K = S.shape
    M = np.zeros(K)
    
    for k in range(K):
        M[k] = np.mean(S[:, :, k])
    
    return M

def calculer_couches_hamiltonien(S, k, alpha, gamma, M):
    """
    Calcule les trois couches de l'hamiltonien pour l'action k
    S: réseau de spins (n x n x K)
    k: indice de l'action
    alpha: paramètre de sensibilité à l'état global
    gamma: coefficient de couplage entre actions
    M: magnétisation de chaque action
    Retourne: trois matrices correspondant aux trois termes de l'hamiltonien
    """
    n, _, K = S.shape
    terme1_layer = np.zeros((n, n))
    terme2_layer = np.zeros((n, n))
    terme3_layer = np.zeros((n, n))
    
    # Précalculer le terme 3 (constant pour tous les sites d'une même action)
    terme3_val = 0
    for k1 in range(K):
        if k1 != k:
            terme3_val -= gamma * M[k1]
    
    terme3_layer.fill(terme3_val)
    
    # Calculer termes 1 et 2 pour chaque site
    for i in range(n):
        for j in range(n):
            # Calcul des indices des 8 voisins avec conditions périodiques
            voisins_i = [(i-1)%n, i, (i+1)%n, (i-1)%n, (i+1)%n, (i-1)%n, i, (i+1)%n]
            voisins_j = [(j-1)%n, (j-1)%n, (j-1)%n, j, j, (j+1)%n, (j+1)%n, (j+1)%n]
            
            # Calcul de la somme des spins des voisins
            somme_voisins = 0
            for v in range(8):
                somme_voisins += S[voisins_i[v], voisins_j[v], k]
            
            # Terme 1: Influence des voisins
            terme1_layer[i, j] = -somme_voisins
            
            # Terme 2: Sensibilité à l'état global du stock k
            terme2_layer[i, j] = alpha * S[i, j, k] * abs(M[k])
    
    return terme1_layer, terme2_layer, terme3_layer

def calculer_hamiltonien_total(terme1_layer, terme2_layer, terme3_layer):
    """
    Calcule l'hamiltonien total à partir des trois couches
    """
    return terme1_layer + terme2_layer + terme3_layer

def mettre_a_jour_spins(S, beta, alpha, gamma, M):
    """
    Met à jour tous les spins selon la dynamique du modèle
    S: réseau de spins actuel (n x n x K)
    beta: paramètre beta = 1/(k_B*T)
    alpha: paramètre de sensibilité à l'état global
    gamma: coefficient de couplage entre actions
    M: magnétisation de chaque action
    Retourne: réseau de spins mis à jour
    """
    n, _, K = S.shape
    S_next = S.copy()
    
    for k in range(K):
        # Pré-calculer les couches de l'hamiltonien pour l'action k
        terme1_layer, terme2_layer, terme3_layer = calculer_couches_hamiltonien(S, k, alpha, gamma, M)
        hamiltonien_total = calculer_hamiltonien_total(terme1_layer, terme2_layer, terme3_layer)
        
        for i in range(n):
            for j in range(n):
                # Calculer l'hamiltonien local
                h_local = hamiltonien_total[i, j]
                
                # Calculer la probabilité de flip
                p = np.exp(-2 * beta * h_local) / (1 + np.exp(-2 * beta * h_local))
                
                # Mettre à jour le spin avec probabilité p
                if np.random.rand() < p:
                    S_next[i, j, k] = 1
                else:
                    S_next[i, j, k] = -1
    
    return S_next

def visualiser_reseau(S, k, ax, titre):
    """
    Visualise le réseau de spins pour l'action k
    """
    # Créer une colormap personnalisée: bleu pour -1, rouge pour +1
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'red'], N=2)
    
    im = ax.imshow(S[:, :, k], cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(titre)
    ax.set_xlabel('j')
    ax.set_ylabel('i')
    return im

def visualiser_hamiltonien(terme1, terme2, terme3, hamiltonien_total, k, fig_size=(16, 4)):
    """
    Visualise les trois couches de l'hamiltonien et l'hamiltonien total
    """
    fig, axes = plt.subplots(1, 4, figsize=fig_size)
    
    # Normaliser pour une meilleure visualisation
    vmin = min(terme1.min(), terme2.min(), terme3.min(), hamiltonien_total.min())
    vmax = max(terme1.max(), terme2.max(), terme3.max(), hamiltonien_total.max())
    
    im1 = axes[0].imshow(terme1, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Action {k+1}: Influence des voisins')
    axes[0].set_xlabel('j')
    axes[0].set_ylabel('i')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(terme2, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Action {k+1}: Sensibilité à l\'état global')
    axes[1].set_xlabel('j')
    axes[1].set_ylabel('i')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(terme3, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Action {k+1}: Couplage inter-actions')
    axes[2].set_xlabel('j')
    axes[2].set_ylabel('i')
    plt.colorbar(im3, ax=axes[2])
    
    im4 = axes[3].imshow(hamiltonien_total, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[3].set_title(f'Action {k+1}: Hamiltonien total')
    axes[3].set_xlabel('j')
    axes[3].set_ylabel('i')
    plt.colorbar(im4, ax=axes[3])
    
    plt.tight_layout()
    return fig

# Programme principal
def main():
    # Créer un dossier pour les images si nécessaire
    output_dir = "ising_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Paramètres
    n = 50          # Taille du réseau (réduite pour la visualisation)
    beta = 2        # Paramètre β = 1/(k_B*T)
    alpha = 0.2     # Sensibilité à l'état global
    gamma = 0.15    # Coefficient de couplage
    K = 2           # Nombre d'actions
    T_steps = 100    # Nombre de mises à jour
    
    print(f"Simulation du modèle d'Ising en éconophysique")
    print(f"Paramètres: n={n}, beta={beta}, alpha={alpha}, gamma={gamma}, K={K}")
    print(f"Nombre d'itérations: {T_steps}")
    
    # Initialisation du réseau de spins
    S = initialiser_reseau(n, K)
    
    # Calcul de la magnétisation initiale
    M = calculer_magnetisation(S)
    print("\nMagnétisations initiales:")
    for k in range(K):
        print(f"Action {k+1}: M = {M[k]:.4f}")
    
    # Visualisation du réseau initial
    fig_initial, axes_initial = plt.subplots(1, K, figsize=(5*K, 5))
    
    # Adapter pour le cas K=1
    if K == 1:
        axes_initial = [axes_initial]
    
    for k in range(K):
        visualiser_reseau(S, k, axes_initial[k], f'Réseau initial (t=0) pour action {k+1}')
    
    plt.tight_layout()
    initial_path = f"{output_dir}/reseau_initial.png"
    fig_initial.savefig(initial_path, dpi=300, bbox_inches='tight')
    plt.close(fig_initial)
    print(f"\nRéseau initial sauvegardé: {initial_path}")
    
    # Simuler l'évolution du système
    print("\nSimulation de l'évolution du système...")
    for t in range(1, T_steps+1):
        # Mettre à jour les spins
        S = mettre_a_jour_spins(S, beta, alpha, gamma, M)
        
        # Calculer la nouvelle magnétisation
        M = calculer_magnetisation(S)
        
        print(f"Itération {t}/{T_steps} complétée")
    
    # Visualisation du réseau final
    fig_final, axes_final = plt.subplots(1, K, figsize=(5*K, 5))
    
    # Adapter pour le cas K=1
    if K == 1:
        axes_final = [axes_final]
    
    for k in range(K):
        visualiser_reseau(S, k, axes_final[k], f'Réseau final (t={T_steps}) pour action {k+1}')
    
    plt.tight_layout()
    final_path = f"{output_dir}/reseau_final.png"
    fig_final.savefig(final_path, dpi=300, bbox_inches='tight')
    plt.close(fig_final)
    print(f"\nRéseau final sauvegardé: {final_path}")
    
    # Afficher les valeurs des magnétisations finales
    print("\nMagnétisations finales:")
    for k in range(K):
        print(f"Action {k+1}: M = {M[k]:.4f}")
    
    # Visualisation des hamiltoniens pour chaque action
    print("\nCalcul et visualisation des hamiltoniens...")
    hamiltonien_paths = []
    for k in range(K):
        terme1, terme2, terme3 = calculer_couches_hamiltonien(S, k, alpha, gamma, M)
        hamiltonien_total = calculer_hamiltonien_total(terme1, terme2, terme3)
        
        # Afficher quelques statistiques sur les hamiltoniens
        print(f"\nStatistiques de l'hamiltonien pour l'action {k+1}:")
        print(f"Terme 1 (Influence des voisins): min={terme1.min():.2f}, max={terme1.max():.2f}, moyenne={terme1.mean():.2f}")
        print(f"Terme 2 (Sensibilité à l'état global): min={terme2.min():.2f}, max={terme2.max():.2f}, moyenne={terme2.mean():.2f}")
        print(f"Terme 3 (Couplage inter-actions): valeur constante = {terme3[0,0]:.2f}")
        print(f"Hamiltonien total: min={hamiltonien_total.min():.2f}, max={hamiltonien_total.max():.2f}, moyenne={hamiltonien_total.mean():.2f}")
        
        fig = visualiser_hamiltonien(terme1, terme2, terme3, hamiltonien_total, k)
        hamiltonien_path = f"{output_dir}/hamiltonien_action_{k+1}.png"
        fig.savefig(hamiltonien_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        hamiltonien_paths.append(hamiltonien_path)
    
    # Résumé des fichiers générés
    print("\nFichiers générés:")
    print(f"1. {initial_path}")
    print(f"2. {final_path}")
    for i, path in enumerate(hamiltonien_paths, 3):
        print(f"{i}. {path}")

if __name__ == "__main__":
    # Fixer la graine aléatoire pour la reproductibilité
    np.random.seed(42)
    main()
