import sys
import os
import pandas as pd
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.intelligence_pipeline import IntelligencePipeline
from src.models.unified_intelligence import UnifiedIntelligence

class TestIntelligencePersistence(unittest.TestCase):
    def setUp(self):
        # Mock Supabase client
        self.mock_supabase = MagicMock()
        self.mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [{'id': 'test-insight-id'}]
        
        # Initialize pipeline with mock client
        self.pipeline = IntelligencePipeline(supabase=self.mock_supabase)
        
        # MOCK DATA
        # 1. Historical Data (df_weekly)
        dates = [datetime.now() - timedelta(days=x*7) for x in range(10)]
        self.df_weekly = pd.DataFrame({
            'week_start': dates,
            'asin': ['TEST_ASIN'] * 10,
            'market_share': [0.1] * 10,
            'price': [20.0] * 10,
            'sales_rank': [1000] * 10,
            'estimated_revenue': [5000.0] * 10,
            'brand': ['TestBrand'] * 10
        })
        
        # 2. Market Snapshot
        self.market_snapshot = {}
        
        # 3. Current Metrics
        self.current_metrics = {
            'asin': 'TEST_ASIN',
            'estimated_monthly_revenue': 20000.0,
            'price': 20.0,
            'sales_rank': 1000,
            'brand': 'TestBrand',
            'category_root': 'Beauty'
        }

    @patch('src.intelligence_pipeline.detect_trigger_events')
    @patch('src.intelligence_pipeline.calculate_revenue_attribution')
    @patch('src.intelligence_pipeline.generate_combined_intelligence')
    def test_full_pipeline_persistence(self, mock_combined, mock_attribution, mock_triggers):
        print("\n🧪 Testing Intelligence Pipeline Full Integration...")
        
        # Setup Mocks
        mock_triggers.return_value = [] # Return simple list of events
        
        # Mock Attribution Output
        from src.revenue_attribution import RevenueAttribution
        mock_attr = MagicMock(spec=RevenueAttribution)
        mock_attr.to_dict.return_value = {"internal": 0, "market": 0}
        mock_attribution.return_value = mock_attr
        
        # Mock Forecasting Output
        from src.predictive_forecasting import CombinedIntelligence
        mock_combined_intel = MagicMock(spec=CombinedIntelligence)
        mock_combined_intel.scenarios = []
        mock_combined_intel.anticipated_events = []
        mock_combined_intel.sustainable_run_rate = 20000.0
        mock_combined.return_value = mock_combined_intel
        
        # Run Pipeline
        # Note: We need to mock _gather_network_intelligence and StrategicTriangulator too
        # checking the code, they are called internally.
        
        with patch.object(self.pipeline, '_gather_network_intelligence', return_value={}), \
             patch('src.intelligence_pipeline.StrategicTriangulator') as MockTriangulator:
            
            # Mock LLM result
            mock_tri = MockTriangulator.return_value
            mock_brief = MagicMock()
            mock_brief.strategic_state = "FORTRESS"
            mock_brief.confidence = 0.9
            mock_brief.reasoning = "Test reasoning"
            mock_brief.thirty_day_risk = 0.0
            mock_brief.thirty_day_growth = 0.0
            mock_brief.ai_recommendation = "Hold steady"
            mock_tri.analyze_strategy.return_value = mock_brief
            
            # EXECUTE
            result = self.pipeline.generate_single_asin_intelligence(
                asin="TEST_ASIN",
                df_weekly=self.df_weekly,
                current_metrics=self.current_metrics,
                market_snapshot=self.market_snapshot
            )
            
            # ASSERTIONS
            self.assertIsNotNone(result)
            print("✅ Pipeline returned result")
            
            # 1. check persistence called
            self.mock_supabase.table.assert_called()
            print("✅ Supabase table accessed")
            
            # 2. Check table names
            calls = [c[0][0] for c in self.mock_supabase.table.call_args_list]
            self.assertIn('strategic_insights', calls)
            print("✅ Written to 'strategic_insights'")
            
            # 3. Check data payload has new fields
            insert_call = self.mock_supabase.table('strategic_insights').insert.call_args
            data = insert_call[0][0]
            
            self.assertIn('revenue_attribution', data)
            self.assertIn('anticipated_events', data)
            self.assertIn('scenarios', data)
            self.assertIn('sustainable_run_rate', data)
            print("✅ Payload contains Phase 2/2.5 fields")

if __name__ == '__main__':
    unittest.main()
