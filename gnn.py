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
from sqlalchemy import create_engine, text
import json
from datetime import datetime

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

def load_data_from_db(engine):
    # Query to get the last 100 flights ordered by date
    query =  text(r"""
        WITH acm_base AS (
            SELECT
                id,
                ac,
                "AIRCRAFTID" AS hl_original, -- Keep original if needed elsewhere, otherwise remove
                regexp_replace("AIRCRAFTID", '\D', '', 'g') AS hl_numeric, -- Calculate numeric hl
                cdate::timestamp without time zone AS cdate,
                "FLTNUMBER" AS flight,
                "timestamp"::timestamp without time zone AS date,
                "EGT1_L",
                "EGT2_L",
                "EGT3_L",
                "EGT4_L",
                "EGT1_R",
                "EGT2_R",
                "EGT3_R",
                "EGT4_R"
            FROM
                b777.b777_acm210
            WHERE
                "timestamp" IS NOT NULL -- Apply date filter early if possible
        ),
        eng_info_base AS (
            SELECT
                hl,
                pos,
                install_date,
                regexp_replace(esn, '\D', '', 'g') AS esn_numeric -- Calculate numeric esn
            FROM
                public.enginfo
        )
        SELECT
            b.id,
            b.ac,
            b.hl_original AS hl, -- Use the original hl format for output
            b.date,
            b.cdate,
            b.flight,
            e.esn_numeric AS esn, -- Use the calculated numeric esn
            -- Select EGT based on engine position
            CASE e.pos
                WHEN '1' THEN b."EGT1_L"
                WHEN '2' THEN b."EGT1_R"
            END AS egt1,
            CASE e.pos
                WHEN '1' THEN b."EGT2_L"
                WHEN '2' THEN b."EGT2_R"
            END AS egt2,
            CASE e.pos
                WHEN '1' THEN b."EGT3_L"
                WHEN '2' THEN b."EGT3_R"
            END AS egt3,
            CASE e.pos
                WHEN '1' THEN b."EGT4_L"
                WHEN '2' THEN b."EGT4_R"
            END AS egt4
        FROM
            acm_base b
        JOIN
            eng_info_base e ON b.hl_numeric = e.hl -- Join on calculated numeric hl
                        AND e.pos IN ('1', '2') -- Match either position
                        AND e.install_date <= b.date -- Match install date condition
        WHERE
            e.esn_numeric IS NOT NULL AND e.esn_numeric <> '' -- Added check for empty string if needed
            AND CASE e.pos
                    WHEN '1' THEN b."EGT1_L" IS NOT NULL AND b."EGT2_L" IS NOT NULL AND b."EGT3_L" IS NOT NULL AND b."EGT4_L" IS NOT NULL
                    WHEN '2' THEN b."EGT1_R" IS NOT NULL AND b."EGT2_R" IS NOT NULL AND b."EGT3_R" IS NOT NULL AND b."EGT4_R" IS NOT NULL
                END
        ORDER BY
            b.date DESC
        LIMIT 500;
    """)
    df = pd.read_sql(query, engine)
    return df

def preprocess_data(df):
    # Extract features and time information
    features = df[['egt1', 'egt2', 'egt3', 'egt4']].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, df[['ac', 'hl', 'esn', 'flight', 'date', 'cdate']].values

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
                        'z_scores': z_scores.tolist(),
                        'context_mean': context_mean.tolist(),
                        'context_std': context_std.tolist()
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

def save_results_to_db(results, engine):
    # Convert context_info to JSON string for database storage
    results['Context_Info'] = results['Context_Info'].apply(lambda x: json.dumps(x) if x is not None else None)
    
    # Save to database
    results.to_sql('anomaly_results', engine, if_exists='replace', index=False, schema="b777")
    print("Results saved to database successfully")

def process_data(model, engine):
    # Load data from database
    df = load_data_from_db(engine)
    
    # Preprocess data
    features, metadata = preprocess_data(df)
    graph_data = create_graph_data(features)
    
    # Detect anomalies
    anomalies, anomaly_scores = detect_anomalies(model, graph_data)
    point, contextual, collective, context_info = classify_anomalies(anomalies, anomaly_scores.numpy(), graph_data.x)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'ac': metadata[:, 0],
        'hl': metadata[:, 1],
        'esn': metadata[:, 2],
        'flight': metadata[:, 3],
        'date': metadata[:, 4],
        'cdate': metadata[:, 5],
        'Anomaly': anomalies.numpy(),
        'Anomaly_Score': anomaly_scores.numpy(),
        'Point_Anomaly': point.numpy(),
        'Contextual_Anomaly': contextual.numpy(),
        'Collective_Anomaly': collective.numpy(),
        'Context_Info': context_info
    })
    
    # Save results to database
    save_results_to_db(results, engine)
    
    return results

# Main execution
if __name__ == "__main__":
    # Database connection
    engine = create_engine('postgresql://postgres:alpha@10.55.32.94:5432/postgres')
    
    # Load and prepare training data
    df = load_data_from_db(engine)
    features, _ = preprocess_data(df)
    graph_data = create_graph_data(features)
    
    # Create data loader
    train_loader = DataLoader([graph_data], batch_size=1, shuffle=True)
    
    # Initialize and train model
    model = GNNAnomalyDetector(num_features=4)  # 4 EGT parameters
    train_model(model, train_loader)
    
    # Process data and save results
    results = process_data(model, engine)
    
    # Print detailed information about contextual anomalies
    for index, row in results[results['Contextual_Anomaly']].iterrows():
        print(f"Contextual Anomaly for ESN {row['esn']}, Flight {row['flight']} at {row['date']}:")
        context = row['Context_Info']
        if context:
            print(f"  Z-scores: {context['z_scores']}")
            print(f"  Context mean: {context['context_mean']}")
            print(f"  Context std: {context['context_std']}")
        print()

    print(results)
