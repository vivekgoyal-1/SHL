from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from typing import List, Dict
import os

# Use the new Google GenAI package (optional - falls back gracefully)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Note: google-generativeai not available. Install with: pip install google-generativeai")

class RecommendationEngineUpdated:
    def __init__(self, gemini_api_key: str = None):
        print("Initializing Recommendation Engine...")
        
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configure Gemini with updated model
        self.llm = None
        if gemini_api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=gemini_api_key)
                # Use the correct model name for the new API
                self.llm = genai.GenerativeModel('gemini-1.5-flash')
                print("✓ Gemini LLM initialized (gemini-1.5-flash)")
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                print("Continuing without LLM enhancement...")
                self.llm = None
        
        # Load assessments
        self.assessments = self._load_assessments()
        self.assessment_embeddings = None
        
        if self.assessments:
            self._create_embeddings()
    
    def _load_assessments(self) -> List[Dict]:
        """Load assessments from JSON file"""
        try:
            # Try enhanced version first
            try:
                with open('data/assessments_enhanced.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
            except FileNotFoundError:
                with open('data/assessments.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
        except FileNotFoundError:
            print("Warning: assessments.json not found. Run scraper first.")
            return []
    
    def _create_embeddings(self):
        """Create embeddings for all assessments"""
        print("Creating embeddings for assessments...")
        
        texts = []
        for assessment in self.assessments:
            # Create rich text representation
            text_parts = []
            
            # Assessment name (most important)
            text_parts.append(assessment['name'])
            
            # Test types with expansion
            if 'test_types' in assessment:
                test_types = assessment['test_types']
                text_parts.append(f"Test types: {test_types}")
                
                # Expand test type meanings
                type_meanings = []
                if 'K' in test_types:
                    type_meanings.append("knowledge skills technical coding programming")
                if 'P' in test_types:
                    type_meanings.append("personality behavior traits temperament")
                if 'A' in test_types:
                    type_meanings.append("ability aptitude cognitive reasoning logical")
                if 'B' in test_types:
                    type_meanings.append("behavioral situational judgment scenarios")
                if 'C' in test_types:
                    type_meanings.append("competency capabilities proficiency")
                if 'S' in test_types:
                    type_meanings.append("simulation practical hands-on exercise")
                
                if type_meanings:
                    text_parts.append(" ".join(type_meanings))
            
            # Description if available
            if 'description' in assessment and assessment['description']:
                text_parts.append(assessment['description'])
            
            text = " ".join(text_parts)
            texts.append(text)
        
        self.assessment_embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Created embeddings for {len(self.assessments)} assessments")
    
    def _enhance_query_with_llm(self, query: str) -> str:
        """Use LLM to extract key skills and requirements from query"""
        if not self.llm:
            # Fallback: simple keyword expansion
            return self._expand_query_keywords(query)
        
        try:
            prompt = f"""Analyze this job hiring query and list the key skills and competencies that need to be assessed.

Job Query: {query}

List only the specific skills, competencies, and traits to assess. Format as comma-separated keywords.
Example: "Java programming, SQL, teamwork, communication, problem-solving"

Keywords:"""
            
            response = self.llm.generate_content(prompt)
            enhanced = response.text.strip()
            return f"{query} {enhanced}"
            
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
            return self._expand_query_keywords(query)
    
    def _expand_query_keywords(self, query: str) -> str:
        """Fallback query expansion without LLM"""
        query_lower = query.lower()
        expansions = []
        
        # Technical skills expansion
        if 'java' in query_lower:
            expansions.append("java programming coding object-oriented")
        if 'python' in query_lower:
            expansions.append("python programming scripting data")
        if 'sql' in query_lower:
            expansions.append("sql database query data")
        if 'javascript' in query_lower:
            expansions.append("javascript web frontend development")
        if 'developer' in query_lower or 'engineer' in query_lower:
            expansions.append("programming coding software technical problem-solving")
        
        # Soft skills expansion
        if 'collaborate' in query_lower or 'team' in query_lower:
            expansions.append("teamwork collaboration interpersonal communication")
        if 'communication' in query_lower:
            expansions.append("communication verbal written presentation")
        if 'leadership' in query_lower or 'lead' in query_lower:
            expansions.append("leadership management motivation delegation")
        if 'customer' in query_lower:
            expansions.append("customer-service interpersonal patience empathy")
        
        # Cognitive/analytical
        if 'analyst' in query_lower or 'analytical' in query_lower:
            expansions.append("analytical reasoning problem-solving critical-thinking data-analysis")
        if 'cognitive' in query_lower:
            expansions.append("cognitive reasoning logical thinking mental-ability")
        if 'personality' in query_lower:
            expansions.append("personality traits behavior temperament work-style")
        
        expanded = " ".join(expansions)
        return f"{query} {expanded}" if expansions else query
    
    def _determine_test_type_needs(self, query: str) -> Dict[str, float]:
        """Determine what test types are needed based on query"""
        query_lower = query.lower()
        
        weights = {
            'K': 0.0,  # Knowledge & Skills
            'P': 0.0,  # Personality & Behavior
            'C': 0.0,  # Competencies
            'B': 0.0,  # Biodata & SJT
            'A': 0.0,  # Ability & Aptitude
            'S': 0.0,  # Simulations
        }
        
        # Technical skills indicators (high weight)
        tech_keywords = ['java', 'python', 'sql', 'javascript', 'coding', 'programming',
                        'developer', 'engineer', 'software', 'technical', '.net', 'c++',
                        'web', 'database', 'api', 'framework']
        if any(kw in query_lower for kw in tech_keywords):
            weights['K'] = 0.6  # Knowledge tests
            weights['A'] = 0.2  # Aptitude tests
            weights['S'] = 0.1  # Simulations
        
        # Soft skills indicators (high weight)
        soft_keywords = ['collaborate', 'collaboration', 'communication', 'team', 'teamwork',
                        'leadership', 'manage', 'interpersonal', 'customer', 'service',
                        'relationship', 'motivate', 'influence']
        if any(kw in query_lower for kw in soft_keywords):
            weights['P'] = 0.6  # Personality tests
            weights['B'] = 0.3  # Behavioral/SJT tests
        
        # Cognitive ability indicators
        cognitive_keywords = ['analytical', 'problem', 'critical thinking', 'logical',
                             'reasoning', 'cognitive', 'mental', 'aptitude', 'ability',
                             'thinking', 'solving']
        if any(kw in query_lower for kw in cognitive_keywords):
            weights['A'] = 0.5  # Ability tests
            weights['C'] = 0.2  # Competency tests
        
        # Explicit mentions
        if 'personality' in query_lower:
            weights['P'] = 0.7
        if 'simulation' in query_lower or 'practical' in query_lower:
            weights['S'] = 0.5
        if 'competenc' in query_lower:
            weights['C'] = 0.5
        
        # Position-based inference
        if 'analyst' in query_lower:
            weights['A'] = 0.4
            weights['K'] = 0.3
        if 'manager' in query_lower or 'supervisor' in query_lower:
            weights['P'] = 0.4
            weights['B'] = 0.3
            weights['C'] = 0.3
        
        return weights
    
    def recommend(self, query: str, top_k: int = 10) -> List[Dict]:
        """Get top K recommendations for a query"""
        
        if not self.assessments:
            return []
        
        # Enhance query
        enhanced_query = self._enhance_query_with_llm(query)
        
        # Encode query
        query_embedding = self.model.encode([enhanced_query])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.assessment_embeddings, query_embedding) / (
            np.linalg.norm(self.assessment_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get test type needs
        test_type_weights = self._determine_test_type_needs(query)
        
        # Adjust scores based on test type matching
        adjusted_scores = []
        for idx, sim in enumerate(similarities):
            test_types = self.assessments[idx].get('test_types', '')
            
            # Calculate type match score
            type_boost = 0.0
            for test_type, weight in test_type_weights.items():
                if test_type in test_types:
                    type_boost += weight
            
            # Combine semantic similarity with type matching
            # 70% semantic, 30% type match
            adjusted_score = (sim * 0.7) + (type_boost * 0.3)
            adjusted_scores.append(adjusted_score)
        
        adjusted_scores = np.array(adjusted_scores)
        
        # Get top candidates (more than needed for balancing)
        top_indices = np.argsort(adjusted_scores)[::-1][:30]
        
        # Build recommendations
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'assessment_name': self.assessments[idx]['name'],
                'url': self.assessments[idx]['url'],
                'score': float(adjusted_scores[idx]),
                'test_types': self.assessments[idx].get('test_types', ''),
                'semantic_score': float(similarities[idx])
            })
        
        # Apply intelligent balancing
        balanced = self._balance_recommendations(query, recommendations, test_type_weights)
        
        # Ensure between 5 and 10
        if len(balanced) < 5:
            # Add more from unbalanced if needed
            for rec in recommendations:
                if rec not in balanced:
                    balanced.append(rec)
                if len(balanced) >= 5:
                    break
        
        return balanced[:10]
    
    def _balance_recommendations(self, query: str, recommendations: List[Dict], 
                                 test_type_weights: Dict[str, float]) -> List[Dict]:
        """Intelligently balance recommendations based on query needs"""
        
        # Identify needed types (those with weight > 0)
        needed_types = [(k, v) for k, v in test_type_weights.items() if v > 0]
        
        if len(needed_types) <= 1:
            # Single type query - just return top scores
            return recommendations[:10]
        
        # Sort needed types by weight
        needed_types.sort(key=lambda x: x[1], reverse=True)
        
        # Group recommendations by primary test type
        by_type = {}
        for rec in recommendations:
            types = rec['test_types']
            # Assign to highest priority matching type
            assigned = False
            for test_type, weight in needed_types:
                if test_type in types:
                    if test_type not in by_type:
                        by_type[test_type] = []
                    by_type[test_type].append(rec)
                    assigned = True
                    break
            
            # If doesn't match any needed type, put in 'other'
            if not assigned:
                if 'other' not in by_type:
                    by_type['other'] = []
                by_type['other'].append(rec)
        
        # Balanced selection
        balanced = []
        total_weight = sum(w for _, w in needed_types)
        
        # Allocate slots proportionally to weights
        for test_type, weight in needed_types:
            if test_type in by_type:
                # Number of slots for this type (minimum 2 if weight > 0)
                slots = max(2, int((weight / total_weight) * 10))
                balanced.extend(by_type[test_type][:slots])
        
        # Fill remaining slots with highest scores
        if len(balanced) < 10:
            remaining = [r for r in recommendations if r not in balanced]
            balanced.extend(remaining[:10 - len(balanced)])
        
        # Re-sort by score
        balanced.sort(key=lambda x: x['score'], reverse=True)
        
        return balanced[:10]