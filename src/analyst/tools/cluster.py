# src/analyst/tools/cluster.py

"""
CLUSTER ENGINE - Product Segmentation Tool

Groups products by behavioral similarity using K-Means clustering.
Identifies:
- Natural product segments
- Behavioral archetypes
- Outlier products

Output: Structured ClusterSignal for the Orchestrator
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..config import KEEPA_CONFIG


@dataclass
class Cluster:
    """A single product cluster."""
    cluster_id: int
    name: str  # Auto-generated descriptive name
    
    # Members
    member_count: int = 0
    member_asins: List[str] = field(default_factory=list)
    
    # Centroid characteristics
    avg_price: Optional[float] = None
    avg_rank: Optional[float] = None
    avg_revenue: Optional[float] = None
    avg_rating: Optional[float] = None
    
    # Behavioral profile
    price_volatility: Optional[float] = None
    rank_trend: str = "STABLE"  # "GROWING", "DECLINING", "STABLE"
    
    # Description
    profile_description: str = ""


@dataclass
class ProductAssignment:
    """Cluster assignment for a single product."""
    asin: str
    cluster_id: int
    cluster_name: str
    distance_to_centroid: float
    is_outlier: bool = False
    outlier_reason: Optional[str] = None


@dataclass
class ClusterSignal:
    """
    Complete clustering output.
    This is what gets passed to the Orchestrator.
    """
    asin: str  # Primary ASIN being analyzed
    analysis_timestamp: str
    
    # Clusters
    clusters: Dict[int, Cluster] = field(default_factory=dict)
    
    # Product assignments
    assignments: Dict[str, ProductAssignment] = field(default_factory=dict)
    
    # Summary
    n_clusters: int = 0
    n_products: int = 0
    n_outliers: int = 0
    
    # Optimal cluster info
    optimal_k: int = 3
    silhouette_score: Optional[float] = None
    
    # Key insights
    cluster_insights: List[str] = field(default_factory=list)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asin": self.asin,
            "timestamp": self.analysis_timestamp,
            "summary": {
                "n_clusters": self.n_clusters,
                "n_products": self.n_products,
                "n_outliers": self.n_outliers,
                "silhouette_score": self.silhouette_score,
            },
            "clusters": {
                str(cid): {
                    "name": c.name,
                    "member_count": c.member_count,
                    "avg_price": c.avg_price,
                    "avg_rank": c.avg_rank,
                    "avg_revenue": c.avg_revenue,
                    "profile": c.profile_description,
                }
                for cid, c in self.clusters.items()
            },
            "assignments": {
                asin: {
                    "cluster_id": a.cluster_id,
                    "cluster_name": a.cluster_name,
                    "is_outlier": a.is_outlier,
                }
                for asin, a in self.assignments.items()
            },
            "insights": self.cluster_insights,
            "warnings": self.warnings,
        }
    
    def to_prompt_string(self) -> str:
        """Format for LLM injection."""
        lines = [f"=== PRODUCT SEGMENTATION ({self.n_products} products) ==="]
        lines.append(f"Found {self.n_clusters} natural clusters, {self.n_outliers} outliers")
        if self.silhouette_score:
            quality = "GOOD" if self.silhouette_score > 0.5 else "MODERATE" if self.silhouette_score > 0.3 else "WEAK"
            lines.append(f"Cluster quality: {quality} (score: {self.silhouette_score:.2f})")
        lines.append("")
        
        lines.append("CLUSTER PROFILES:")
        for cid, cluster in sorted(self.clusters.items()):
            lines.append(f"  [{cid}] {cluster.name} ({cluster.member_count} products)")
            lines.append(f"      Avg Price: ${cluster.avg_price:.2f}, Avg Rank: {cluster.avg_rank:.0f}")
            lines.append(f"      Trend: {cluster.rank_trend}")
            if cluster.profile_description:
                lines.append(f"      → {cluster.profile_description}")
        
        # Show where the primary ASIN falls
        if self.asin in self.assignments:
            assignment = self.assignments[self.asin]
            lines.append("")
            lines.append(f"YOUR PRODUCT ({self.asin}):")
            lines.append(f"  Cluster: {assignment.cluster_name}")
            if assignment.is_outlier:
                lines.append(f"  ⚠️ OUTLIER: {assignment.outlier_reason}")
        
        if self.cluster_insights:
            lines.append("")
            lines.append("INSIGHTS:")
            for insight in self.cluster_insights:
                lines.append(f"  • {insight}")
        
        if self.warnings:
            lines.append("")
            # Defensive: ensure all warnings are strings
            warn_strs = [str(w) if not isinstance(w, str) else w for w in self.warnings]
            lines.append("WARNINGS: " + " | ".join(warn_strs))
        
        return "\n".join(lines)


def segment_products(
    df_summary: pd.DataFrame,
    primary_asin: str = "UNKNOWN",
    features: Optional[List[str]] = None,
    max_clusters: int = 5,
    min_cluster_size: int = 3
) -> ClusterSignal:
    """
    Segment products into behavioral clusters.
    
    Args:
        df_summary: Summary data with one row per product
        primary_asin: The main ASIN being analyzed
        features: Features to use for clustering
        max_clusters: Maximum number of clusters to consider
        min_cluster_size: Minimum products per cluster
        
    Returns:
        ClusterSignal with segmentation results
    """
    signal = ClusterSignal(
        asin=primary_asin,
        analysis_timestamp=datetime.now().isoformat()
    )
    
    if df_summary is None or len(df_summary) < 5:
        signal.warnings.append("INSUFFICIENT_DATA_FOR_CLUSTERING")
        return signal
    
    signal.n_products = len(df_summary)
    
    # Default features for clustering
    if features is None:
        features = [
            "filled_price", "sales_rank", "weekly_revenue",
            "rating", "review_count", "new_offer_count"
        ]
    
    # Filter to available features
    features = [f for f in features if f in df_summary.columns]
    
    if len(features) < 2:
        signal.warnings.append("INSUFFICIENT_FEATURES")
        return signal
    
    # Prepare data
    df = df_summary.copy()
    
    # Ensure ASIN column exists
    if 'asin' not in df.columns:
        df['asin'] = df.index.astype(str)
    
    try:
        # Extract feature matrix
        X, valid_asins = _prepare_features(df, features)
        
        if len(X) < 5:
            signal.warnings.append("TOO_FEW_VALID_PRODUCTS")
            return signal
        
        # Find optimal number of clusters
        optimal_k, silhouette = _find_optimal_k(X, max_clusters, min_cluster_size)
        signal.optimal_k = optimal_k
        signal.silhouette_score = silhouette
        signal.n_clusters = optimal_k
        
        # Run K-Means
        labels, centroids = _run_kmeans(X, optimal_k)
        
        # Build clusters
        signal.clusters = _build_clusters(df, valid_asins, labels, features)
        
        # Assign products
        signal.assignments = _assign_products(df, valid_asins, labels, centroids, X, signal.clusters)
        
        # Count outliers
        signal.n_outliers = sum(1 for a in signal.assignments.values() if a.is_outlier)
        
        # Generate insights
        signal.cluster_insights = _generate_cluster_insights(signal, primary_asin)
        
    except Exception as e:
        signal.warnings.append(f"CLUSTERING_FAILED: {str(e)}")
    
    return signal


def _prepare_features(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Prepare and normalize feature matrix."""
    from sklearn.preprocessing import StandardScaler
    
    # Get valid rows (no NaN in features)
    valid_mask = df[features].notna().all(axis=1)
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) < 5:
        raise ValueError("Too few valid products")
    
    X = valid_df[features].values.astype(float)
    
    # Handle potential log transform for rank (highly skewed)
    if 'sales_rank' in features:
        idx = features.index('sales_rank')
        X[:, idx] = np.log1p(X[:, idx])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get ASINs
    asins = valid_df['asin'].tolist() if 'asin' in valid_df.columns else valid_df.index.tolist()
    
    return X_scaled, [str(a) for a in asins]


def _find_optimal_k(X: np.ndarray, max_k: int, min_size: int) -> Tuple[int, float]:
    """Find optimal number of clusters using silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    n_samples = len(X)
    max_k = min(max_k, n_samples // min_size)
    
    if max_k < 2:
        return 2, 0.0
    
    best_k = 2
    best_score = -1
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Check minimum cluster size
        unique, counts = np.unique(labels, return_counts=True)
        if min(counts) < min_size:
            continue
        
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k, float(best_score)


def _run_kmeans(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run K-Means clustering."""
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    return labels, centroids


def _build_clusters(
    df: pd.DataFrame,
    asins: List[str],
    labels: np.ndarray,
    features: List[str]
) -> Dict[int, Cluster]:
    """Build cluster objects with profiles."""
    clusters = {}
    
    asin_col = 'asin' if 'asin' in df.columns else df.index.name or 'index'
    
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        member_asins = [asins[i] for i, m in enumerate(mask) if m]
        
        # Get member data
        if asin_col == 'index':
            member_df = df[df.index.isin(member_asins)]
        else:
            member_df = df[df[asin_col].astype(str).isin(member_asins)]
        
        cluster = Cluster(
            cluster_id=int(cluster_id),
            name=f"Cluster_{cluster_id}",
            member_count=len(member_asins),
            member_asins=member_asins
        )
        
        # Calculate averages
        if 'filled_price' in member_df.columns:
            cluster.avg_price = float(member_df['filled_price'].mean())
        if 'sales_rank' in member_df.columns:
            cluster.avg_rank = float(member_df['sales_rank'].mean())
        if 'weekly_revenue' in member_df.columns:
            cluster.avg_revenue = float(member_df['weekly_revenue'].mean())
        if 'rating' in member_df.columns:
            cluster.avg_rating = float(member_df['rating'].mean())
        
        # Generate name based on characteristics
        cluster.name = _generate_cluster_name(cluster)
        cluster.profile_description = _generate_cluster_profile(cluster, member_df)
        
        clusters[cluster_id] = cluster
    
    return clusters


def _generate_cluster_name(cluster: Cluster) -> str:
    """Generate a descriptive name for the cluster."""
    parts = []
    
    # Price tier
    if cluster.avg_price:
        if cluster.avg_price > 30:
            parts.append("Premium")
        elif cluster.avg_price > 15:
            parts.append("Mid-Range")
        else:
            parts.append("Budget")
    
    # Performance tier
    if cluster.avg_rank:
        if cluster.avg_rank < 1000:
            parts.append("Top-Sellers")
        elif cluster.avg_rank < 10000:
            parts.append("Strong-Performers")
        elif cluster.avg_rank < 50000:
            parts.append("Mid-Tier")
        else:
            parts.append("Long-Tail")
    
    if parts:
        return " ".join(parts)
    return f"Cluster_{cluster.cluster_id}"


def _generate_cluster_profile(cluster: Cluster, member_df: pd.DataFrame) -> str:
    """Generate a description of the cluster."""
    descriptions = []
    
    if cluster.avg_price:
        descriptions.append(f"${cluster.avg_price:.0f} avg price")
    
    if cluster.avg_rank:
        descriptions.append(f"Rank {cluster.avg_rank:.0f} avg")
    
    if cluster.avg_rating:
        descriptions.append(f"{cluster.avg_rating:.1f} stars")
    
    # Check for any standout characteristics
    if 'new_offer_count' in member_df.columns:
        avg_sellers = member_df['new_offer_count'].mean()
        if avg_sellers > 10:
            descriptions.append("high competition")
        elif avg_sellers < 3:
            descriptions.append("low competition")
    
    return ", ".join(descriptions) if descriptions else "Standard profile"


def _assign_products(
    df: pd.DataFrame,
    asins: List[str],
    labels: np.ndarray,
    centroids: np.ndarray,
    X: np.ndarray,
    clusters: Dict[int, Cluster]
) -> Dict[str, ProductAssignment]:
    """Assign each product to a cluster and detect outliers."""
    assignments = {}
    
    for i, asin in enumerate(asins):
        cluster_id = int(labels[i])
        
        # Calculate distance to centroid
        distance = np.linalg.norm(X[i] - centroids[cluster_id])
        
        # Calculate typical distance for this cluster
        cluster_mask = labels == cluster_id
        cluster_distances = [
            np.linalg.norm(X[j] - centroids[cluster_id])
            for j in range(len(X)) if cluster_mask[j]
        ]
        mean_distance = np.mean(cluster_distances)
        std_distance = np.std(cluster_distances)
        
        # Flag as outlier if more than 2 std away
        is_outlier = distance > mean_distance + 2 * std_distance
        outlier_reason = None
        if is_outlier:
            outlier_reason = f"Distance {distance:.2f} > threshold {mean_distance + 2*std_distance:.2f}"
        
        assignments[asin] = ProductAssignment(
            asin=asin,
            cluster_id=cluster_id,
            cluster_name=clusters[cluster_id].name if cluster_id in clusters else f"Cluster_{cluster_id}",
            distance_to_centroid=float(distance),
            is_outlier=is_outlier,
            outlier_reason=outlier_reason
        )
    
    return assignments


def _generate_cluster_insights(signal: ClusterSignal, primary_asin: str) -> List[str]:
    """Generate strategic insights from clustering."""
    insights = []
    
    # Insight 1: Market structure
    if signal.clusters:
        price_ranges = [(c.cluster_id, c.avg_price) for c in signal.clusters.values() if c.avg_price]
        if len(price_ranges) >= 2:
            prices = [p for _, p in price_ranges]
            price_spread = max(prices) - min(prices)
            if price_spread > 20:
                insights.append(f"Market shows ${price_spread:.0f} price spread across segments")
    
    # Insight 2: Competition concentration
    if signal.clusters:
        sizes = [c.member_count for c in signal.clusters.values()]
        largest = max(sizes)
        if largest > signal.n_products * 0.5:
            insights.append(f"Concentrated market: largest cluster has {largest/signal.n_products:.0%} of products")
    
    # Insight 3: Where does the primary ASIN fall?
    if primary_asin in signal.assignments:
        assignment = signal.assignments[primary_asin]
        cluster = signal.clusters.get(assignment.cluster_id)
        if cluster:
            insights.append(f"Your product is in the '{cluster.name}' segment")
            if assignment.is_outlier:
                insights.append("Your product is an outlier within its cluster - potential for differentiation")
    
    # Insight 4: Opportunity identification
    for cluster in signal.clusters.values():
        if cluster.avg_rank and cluster.avg_rank < 5000 and cluster.member_count < 5:
            insights.append(f"'{cluster.name}' is a small but high-performing segment - potential opportunity")
    
    return insights
