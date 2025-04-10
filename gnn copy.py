#GNN Fault Detection
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

class GNNAnomalyDetector(torch.nn.Module):
    def __init__(self, num_features):
        super(GNNAnomalyDetector, self).__init__()
        self.conv1 = GCNConv(num_features, 32)  
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 8)
        self.conv4 = GCNConv(8, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    features = df[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def create_graph_data(features):
    num_nodes = features.shape[0]
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i+1])  
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def train_model(model, train_loader, num_epochs=100, accumulation_steps=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, data in enumerate(train_loader):
            out = model(data)
            loss = F.mse_loss(out, data.x)
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

def detect_anomalies(model, data, threshold=2.5):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
    mse = F.mse_loss(reconstructed, data.x, reduction='none').mean(dim=1)
    anomaly_scores = (mse - mse.mean()) / mse.std()
    anomalies = anomaly_scores > threshold
    return anomalies, anomaly_scores

def analyze_context(data, index, window_size):
    start = max(0, index - window_size)
    end = min(len(data), index + window_size + 1)
    window = data[start:end]
    context_mean = np.mean(window, axis=0)
    context_std = np.std(window, axis=0)
    z_scores = (data[index] - context_mean) / context_std
    return z_scores, context_mean, context_std

def classify_anomalies(anomalies, anomaly_scores, data, window_size=5):
    point_anomalies = anomalies.clone()
    contextual_anomalies = torch.zeros_like(anomalies, dtype=torch.bool)
    collective_anomalies = torch.zeros_like(anomalies, dtype=torch.bool)
    context_info = []

    for i in range(len(anomalies)):
        if anomalies[i]:
            if i < window_size or i >= len(anomalies) - window_size:
                point_anomalies[i] = True
                context_info.append(None)
            else:
                z_scores, context_mean, context_std = analyze_context(data.numpy(), i, window_size)
                if np.any(np.abs(z_scores) > 2):  
                    contextual_anomalies[i] = True
                    point_anomalies[i] = False
                    context_info.append({
                        'z_scores': z_scores,
                        'context_mean': context_mean,
                        'context_std': context_std
                    })
                else:
                    point_anomalies[i] = True
                    context_info.append(None)
        else:
            context_info.append(None)

    for i in range(len(anomalies) - window_size):
        if torch.all(anomalies[i:i+window_size]):
            collective_anomalies[i:i+window_size] = True
            point_anomalies[i:i+window_size] = False
            contextual_anomalies[i:i+window_size] = False

    return point_anomalies, contextual_anomalies, collective_anomalies, context_info

def plot_anomalies(time, anomalies, anomaly_scores, point_anomalies, contextual_anomalies, collective_anomalies):
    plt.figure(figsize=(14, 7))
    plt.plot(time, anomaly_scores, label='Anomaly Scores', color='blue')
    plt.scatter(time[point_anomalies], anomaly_scores[point_anomalies], color='green', label='Point Anomalies')
    plt.scatter(time[contextual_anomalies], anomaly_scores[contextual_anomalies], color='orange', label='Contextual Anomalies')
    plt.scatter(time[collective_anomalies], anomaly_scores[collective_anomalies], color='purple', label='Collective Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.title('Anomaly Detection Results')
    plt.show()

# Save anomalies to CSV after detection
def save_anomalies_to_csv(results, output_file="anomalies_detected.csv"):
    results.to_csv(output_file, index=False)
    print(f"Anomalies saved to {output_file}")

def process_new_datasets(file_paths, model, output_file="anomalies_detected.csv"):
    all_results = []
    
    for file_path in file_paths:
        features = load_and_preprocess_data(file_path)
        graph_data = create_graph_data(features)
        anomalies, anomaly_scores = detect_anomalies(model, graph_data)
        point, contextual, collective, context_info = classify_anomalies(anomalies, anomaly_scores.numpy(), graph_data.x)
        
        df = pd.read_csv(file_path)
        results = pd.DataFrame({
            'Time': df['TIME'],
            'Anomaly': anomalies.numpy(),
            'Anomaly_Score': anomaly_scores.numpy(),
            'Point_Anomaly': point.numpy(),
            'Contextual_Anomaly': contextual.numpy(),
            'Collective_Anomaly': collective.numpy(),
            'Context_Info': context_info
        })
        
        # Plot anomalies
        plot_anomalies(
            time=df['TIME'].values,
            anomalies=results['Anomaly'].values,
            anomaly_scores=results['Anomaly_Score'].values,
            point_anomalies=results['Point_Anomaly'].values,
            contextual_anomalies=results['Contextual_Anomaly'].values,
            collective_anomalies=results['Collective_Anomaly'].values
        )
        
        all_results.append(results)
    
    # Concatenate all results into a single DataFrame
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save the anomalies to a CSV file
    save_anomalies_to_csv(final_results, output_file)
    
    return final_results

# Main execution
if __name__ == "__main__":
    # Training
    all_data = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(script_dir, 'Training')
    
    # Get list of all CSV files in the Training directory
    existing_files = [f for f in os.listdir(training_dir) if f.startswith('Case') and f.endswith('.csv')]
    print(f"Found {len(existing_files)} training files")
    
    for file_name in existing_files:
        try:
            file_path = os.path.join(training_dir, file_name)
            print(f"Processing {file_name}...")
            features = load_and_preprocess_data(file_path)
            graph_data = create_graph_data(features)
            all_data.append(graph_data)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

    if not all_data:
        raise ValueError("No valid training data found. Please check the Training directory.")

    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = GNNAnomalyDetector(num_features=7)
    train_model(model, train_loader)

    # Testing
    testing_dir = os.path.join(script_dir, 'Testing')
    file_paths = [os.path.join(testing_dir, f'test_dataset{i}.csv') for i in range(1, 11)]
    all_results = process_new_datasets(file_paths, model)

    # Print detailed information about contextual anomalies
    for index, row in all_results[all_results['Contextual_Anomaly']].iterrows():
        print(f"Contextual Anomaly at time {row['Time']}:")
        context = row['Context_Info']
        if context:
            print(f"  Z-scores: {context['z_scores']}")
            print(f"  Context mean: {context['context_mean']}")
            print(f"  Context std: {context['context_std']}")
        print()

    print(all_results)
