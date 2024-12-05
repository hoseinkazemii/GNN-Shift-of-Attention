# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import time

# # Assuming eye_df is already returned and available
# def plot_gaze_frequency_heatmap(eye_df):
#     # Filter the data to only include gaze fixations (or focus events)
#     gaze_counts = eye_df['Name'].value_counts()
    
#     # Get the top 10 objects sorted in descending order of frequency
#     top_10_gaze_counts = gaze_counts.head(10).sort_values(ascending=False)
    
#     # Convert to DataFrame for easier plotting
#     gaze_counts_df = pd.DataFrame({'Object': top_10_gaze_counts.index, 'Frequency': top_10_gaze_counts.values})
    
#     # Sort the DataFrame by frequency in descending order
#     gaze_counts_df = gaze_counts_df.sort_values(by='Frequency', ascending=False)
    
#     # Pivot the data to create a matrix suitable for heatmap visualization
#     gaze_pivot = gaze_counts_df.pivot_table(index='Object', values='Frequency')
    
#     # Plot the heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(gaze_pivot, annot=True, cmap='YlGnBu', linewidths=.5, fmt='g')
#     plt.title('Heatmap of Gaze Frequency on Top 10 Objects')
#     plt.ylabel('Object')
#     plt.tight_layout()
#     plt.savefig("Heatmap of Gaze Frequency on Top 10 Objects.png", format="png", dpi=300)
#     plt.show()

# # Call the function to plot the heatmap
# plot_gaze_frequency_heatmap(eye_df)  # Use either eye_df or eye_df_filtered based on what you want to analyze


# # Plot Time-Series Line Plot of Gaze on PowerLine1
# def plot_gaze_on_powerline(eye_df):
#     # Create a Fixation Duration column by calculating the difference between consecutive timeframes for each object
#     eye_df['Fixation Duration'] = eye_df.groupby('Name')['Timeframe'].diff().fillna(0)
    
#     # Filter the data to only include gaze fixations on PowerLine1
#     powerline_df = eye_df[eye_df['Name'] == 'PowerLine1']
    
#     # Plot the time-series line plot
#     plt.figure(figsize=(12, 6))
#     plt.plot(powerline_df['Timeframe'], powerline_df['Fixation Duration'], label='Gaze on PowerLine1', color='b')
#     plt.xlabel('Timeframe')
#     plt.ylabel('Fixation Duration')
#     plt.title('Time-Series Line Plot of Gaze on PowerLine1')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("Time-Series Line Plot of Gaze on PowerLine1.png", format="png", dpi=300)
#     plt.show()

# # Call the function to plot the time-series line plot
# plot_gaze_on_powerline(eye_df)  # Use either eye_df or eye_df_filtered based on what you want to analyze


# # Plot Gaze Path Visualization with Arrows Showing Transitions
# def plot_gaze_path(eye_df):
#     # Create a directed graph to visualize the transitions
#     G = nx.DiGraph()
    
#     # Iterate through the data to create nodes and edges
#     for i in range(1, len(eye_df)):
#         current_object = eye_df.iloc[i]['Name']
#         previous_object = eye_df.iloc[i - 1]['Name']
#         if previous_object != current_object:
#             if G.has_edge(previous_object, current_object):
#                 G[previous_object][current_object]['weight'] += 1
#             else:
#                 G.add_edge(previous_object, current_object, weight=1)
    
#     # Position nodes using spring layout for better visualization
#     pos = nx.spring_layout(G, seed=42)
    
#     # Draw nodes
#     plt.figure(figsize=(12, 8))
#     nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
    
#     # Draw edges with arrows, using weights to determine width
#     edges = G.edges(data=True)
#     nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in edges], arrowstyle='-|>', arrowsize=20, edge_color='gray', width=[d['weight'] * 0.1 for u, v, d in edges])
    
#     # Highlight paths that lead to 'PowerLine1'
#     if 'PowerLine1' in G:
#         edges_to_powerline = [(u, v) for u, v in G.in_edges('PowerLine1')]
#         nx.draw_networkx_edges(G, pos, edgelist=edges_to_powerline, edge_color='red', width=2.0, arrowstyle='-|>', arrowsize=20)
    
#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
#     plt.title('Gaze Path Visualization with Transitions to PowerLine1 Highlighted')
#     plt.tight_layout()
#     plt.savefig("Gaze Path Visualization with Transitions to PowerLine1 Highlighted.png", format="png", dpi=300)
#     plt.show()

# # Call the function to plot the gaze path
# plot_gaze_path(eye_df_filtered)  # Use either eye_df or eye_df_filtered based on what you want to analyze


# # Plot Gaze Path Visualization with Arrows Showing Transitions Incrementally
# def plot_gaze_path_incrementally(eye_df):
#     # Create a directed graph to visualize the transitions
#     G = nx.DiGraph()
#     node_sizes = {}
    
#     # Iterate through the data incrementally and visualize each shift of attention
#     for i in range(1, len(eye_df)):
#         current_object = eye_df.iloc[i]['Name']
#         previous_object = eye_df.iloc[i - 1]['Name']
        
#         # If there's a shift in attention, update the graph
#         if previous_object != current_object:
#             if G.has_edge(previous_object, current_object):
#                 G[previous_object][current_object]['weight'] += 1
#             else:
#                 G.add_edge(previous_object, current_object, weight=1)
            
#             # Update node size to indicate repeated fixations
#             if current_object in node_sizes:
#                 node_sizes[current_object] += 300
#             else:
#                 node_sizes[current_object] = 700
            
#             # Position nodes using spring layout for better visualization
#             pos = nx.spring_layout(G, seed=42)
            
#             # Draw nodes
#             plt.figure(figsize=(12, 8))
#             nx.draw_networkx_nodes(G, pos, node_size=[node_sizes.get(node, 700) for node in G.nodes], node_color='lightblue')
            
#             # Draw edges with arrows, using weights to determine width
#             edges = G.edges(data=True)
#             edge_colors = []
#             for u, v, d in edges:
#                 if (u == previous_object) and (v == current_object):
#                     edge_colors.append('blue')  # Highlight the most recent edge in blue
#                 else:
#                     edge_colors.append('gray')
            
#             nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in edges], arrowstyle='-|>', arrowsize=20, edge_color=edge_colors, width=[d['weight'] * 0.1 for u, v, d in edges])
            
#             # Highlight paths that lead to 'PowerLine1'
#             if 'PowerLine1' in G:
#                 edges_to_powerline = [(u, v) for u, v in G.in_edges('PowerLine1')]
#                 nx.draw_networkx_edges(G, pos, edgelist=edges_to_powerline, edge_color='red', width=2.0, arrowstyle='-|>', arrowsize=20)
            
#             # Draw labels
#             nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
            
#             plt.title(f'Gaze Path Visualization after Shift of Attention to {current_object}')
#             plt.tight_layout()
#             plt.pause(10)
#             plt.close()

# # Call the function to plot the gaze path incrementally
# plot_gaze_path_incrementally(eye_df_filtered)  # Use either eye_df or eye_df_filtered based on what you want to analyze


# # Plot Object Proximity vs. Attention Probability (Scatter Plot)
# def plot_object_proximity_vs_attention(eye_df):
#     # Calculate the proximity to each object and the attention probability
#     proximity_data = []
#     attention_data = eye_df['Name'].value_counts(normalize=True)
    
#     # Assuming 'Object X', 'Object Y', 'Object Z' columns represent the object's position
#     for obj_name in eye_df['Name'].unique():
#         obj_df = eye_df[eye_df['Name'] == obj_name]
#         if len(obj_df) > 0:
#             mean_proximity = np.mean(
#                 (obj_df['Object X']**2 + obj_df['Object Y']**2 + obj_df['Object Z']**2) ** 0.5
#             )
#             proximity_data.append((obj_name, mean_proximity, attention_data[obj_name]))
    
#     # Convert proximity data to a DataFrame for easier plotting
#     proximity_df = pd.DataFrame(proximity_data, columns=['Object', 'Proximity', 'Attention Probability'])
    
#     # Plot the scatter plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(proximity_df['Proximity'], proximity_df['Attention Probability'], alpha=0.7, color='b')
#     plt.xlabel('Proximity to Object')
#     plt.ylabel('Attention Probability')
#     plt.title('Object Proximity vs. Attention Probability')
#     plt.tight_layout()
#     plt.show()

# # Call the function to plot object proximity vs attention probability
# plot_object_proximity_vs_attention(eye_df_filtered)  # Use either eye_df or eye_df_filtered based on what you want to analyze


# # Analyze shifts of attention to PowerLine1 and plot histogram of distances
# def analyze_attention_shifts(eye_df):
#     shifts_to_powerline = []
#     distances = []
    
#     for i in range(1, len(eye_df)):
#         current_object = eye_df.iloc[i]['Name']
#         previous_object = eye_df.iloc[i - 1]['Name']
        
#         if current_object == 'PowerLine1' and previous_object != current_object:
#             shifts_to_powerline.append((previous_object, current_object))
            
#             # Calculate distance between previous object and PowerLine1
#             distance = np.sqrt(
#                 (eye_df.iloc[i]['Object X'] - eye_df.iloc[i - 1]['Object X']) ** 2 +
#                 (eye_df.iloc[i]['Object Y'] - eye_df.iloc[i - 1]['Object Y']) ** 2 +
#                 (eye_df.iloc[i]['Object Z'] - eye_df.iloc[i - 1]['Object Z']) ** 2
#             )
#             distances.append(distance)
    
#     # Plot histogram of distances
#     plt.figure(figsize=(10, 6))
#     plt.hist(distances, bins=20, color='g', alpha=0.7)
#     plt.xlabel('Distance to PowerLine1')
#     plt.ylabel('Number of Shifts of Attention')
#     plt.title('Histogram of Distances for Shifts of Attention to PowerLine1')
#     plt.tight_layout()
#     plt.show()

# # Call the function to analyze attention shifts and plot histogram
# analyze_attention_shifts(eye_df)  # Use either eye_df or eye_df_filtered based on what you want to analyze


# # Plot Histogram of Shift of Attention vs Distance Ranges
# def plot_shift_vs_distance_range(eye_df):
#     distances = []
    
#     # Calculate distances for each shift of attention
#     for i in range(1, len(eye_df)):
#         current_object = eye_df.iloc[i]['Name']
#         previous_object = eye_df.iloc[i - 1]['Name']
        
#         if previous_object != current_object:
#             distance = np.sqrt(
#                 (eye_df.iloc[i]['Object X'] - eye_df.iloc[i - 1]['Object X']) ** 2 +
#                 (eye_df.iloc[i]['Object Y'] - eye_df.iloc[i - 1]['Object Y']) ** 2 +
#                 (eye_df.iloc[i]['Object Z'] - eye_df.iloc[i - 1]['Object Z']) ** 2
#             )
#             distances.append(distance)
    
#     # Determine the maximum distance and create bins in 30m ranges
#     max_distance = max(distances) if distances else 0
#     bins = list(range(0, int(max_distance) + 30, 30))
#     bins_labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)]
#     bins_labels.append(f'{bins[-1]}+')
    
#     # Allocate distances to appropriate bins
#     distance_ranges = {label: 0 for label in bins_labels}
#     for distance in distances:
#         for i in range(len(bins) - 1):
#             if bins[i] <= distance < bins[i + 1]:
#                 distance_ranges[bins_labels[i]] += 1
#                 break
#         else:
#             distance_ranges[bins_labels[-1]] += 1
    
#     # Plot histogram of distance ranges
#     plt.figure(figsize=(10, 6))
#     plt.bar(distance_ranges.keys(), distance_ranges.values(), color='b', alpha=0.7)
#     plt.xlabel('Distance Range Between Objects (m)')
#     plt.ylabel('Number of Shifts of Attention')
#     plt.title('Histogram of Shifts of Attention vs Distance Ranges')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("Histogram of Shifts of Attention vs Distance Ranges.png", format="png", dpi=300)
#     plt.show()

# # Call the function to plot shift of attention vs distance ranges
# plot_shift_vs_distance_range(eye_df)  # Use either eye_df or eye_df_filtered based on what you want to analyze










# # Plotting attention shift comparison for different objects
# def plot_attention_shifts(df):
#     # Count the occurrences of each object being fixated
#     fixation_counts = df['Name'].value_counts()
    
#     # Plot the fixation counts
#     plt.figure(figsize=(10, 6))
#     fixation_counts.plot(kind='bar', color='skyblue')
#     plt.xlabel('Objects')
#     plt.ylabel('Number of Fixations')
#     plt.title('Attention Shifts to Different Objects')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()

# Call the plotting function
# plot_attention_shifts(eye_tracking_data)
# Plotting attention shift comparison for different objects towards the powerline
# def plot_attention_shifts_to_powerline(df):
#     # Identify shifts in attention by checking changes in the 'Name' column
#     df['Previous_Name'] = df['Name'].shift(1)
#     df['Attention_Shift'] = (df['Name'] != df['Previous_Name'])
    
#     # Set the first row's 'Attention_Shift' to False since it cannot be a shift
#     df.loc[0, 'Attention_Shift'] = False
    
#     # Filter rows where there is an attention shift to 'PowerLine1'
#     powerline_shifts = df[(df['Name'] == 'PowerLine1') & (df['Attention_Shift'])]
    
#     # Count the occurrences of each object shifting attention to the powerline
#     shift_counts = powerline_shifts['Previous_Name'].value_counts()

#     # Plot the shift counts
#     plt.figure(figsize=(10, 6))
#     shift_counts.plot(kind='bar', color='skyblue')
#     plt.xlabel('Objects')
#     plt.ylabel('Number of Attention Shifts to PowerLine1')
#     plt.title('Attention Shifts to PowerLine1 from Different Objects')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig('Attention Shifts to PowerLine1 from Different Objects.png')
#     plt.show()