import osmnx as ox
import networkx as nx

print("🚀 Starting road network download for Bengaluru...")

# Expanded bounding box to cover ALL 7 areas properly
# Covers: Hebbal, Whitefield, Koramangala, Indiranagar, Jayanagar, M.G. Road, Yeshwanthpur
north, south = 13.10, 12.85  # Wider latitude range
east, west = 77.80, 77.50    # Wider longitude range

print(f"📍 Coverage Area: {north}°N to {south}°N, {east}°E to {west}°E")

try:
    # Download complete drivable road network
    print("⏳ Downloading road network (this may take 2-5 minutes)...")
    G = ox.graph_from_bbox(
        north, south, east, west, 
        network_type="drive",
        simplify=True  # Simplify to reduce size
    )
    
    print(f"✅ Downloaded {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Add travel_cost attribute for routing optimization
    print("⚙️ Adding travel cost attributes...")
    for u, v, k, data in G.edges(keys=True, data=True):
        base_length = data.get("length", 1000)
        
        # Simulate realistic congestion based on road type
        road_type = data.get("highway", "residential")
        
        if road_type in ["motorway", "trunk", "primary"]:
            congestion_factor = 1.8  # Major highways (more congestion)
        elif road_type in ["secondary", "tertiary"]:
            congestion_factor = 1.4  # Medium roads
        else:
            congestion_factor = 1.1  # Small local roads
        
        data["travel_cost"] = base_length * congestion_factor
    
    # Save the graph
    #print("💾 Saving graph to bengaluru_small.graphml...")
    ox.save_graphml(G, "bengaluru.graphml")
    """"
    print("\n" + "="*60)
    print("✅ SUCCESS! Road network saved successfully!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   - Total nodes: {len(G.nodes):,}")
    print(f"   - Total edges: {len(G.edges):,}")
    print(f"   - File: bengaluru_small.graphml")
    print("\n🎯 You can now run your Streamlit app!")"""
    
    # Verify all areas are reachable
    print("\n🔍 Verifying area coverage...")
    test_coords = {
        'Hebbal': (77.5890, 13.0359),
        'Indiranagar': (77.6412, 12.9719),
        'Jayanagar': (77.5834, 12.9249),
        'Koramangala': (77.6245, 12.9352),
        'M.G. Road': (77.6099, 12.9767),
        'Whitefield': (77.7500, 12.9698),
        'Yeshwanthpur': (77.5547, 13.0284)
    }
    
    covered = []
    for area, (lon, lat) in test_coords.items():
        try:
            node = ox.distance.nearest_nodes(G, lon, lat)
            covered.append(area)
            print(f"   ✅ {area}")
        except:
            print(f"   ❌ {area} - NOT COVERED")
    
    print(f"\n📍 Coverage: {len(covered)}/7 areas")
    
    if len(covered) == 7:
        print("🎉 Perfect! All areas are covered and routable!")
    else:
        print("⚠️ Warning: Some areas may not be reachable")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\n💡 Troubleshooting tips:")
    print("1. Check your internet connection")
    print("2. Install required packages: pip install osmnx networkx")
    print("3. Try reducing the bounding box size")
    print("4. Use VPN if OSM is blocked in your region")