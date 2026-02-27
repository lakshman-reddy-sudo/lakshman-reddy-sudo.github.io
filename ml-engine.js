/**
 * Client-Side ML Engine for SymptomAI
 * Implements TF-IDF + Cosine Similarity for symptom classification
 * entirely in the browser — no backend required.
 */

class SymptomMLEngine {
    constructor() {
        this.trainingData = [];
        this.diseaseInfo = {};
        this.vocabulary = {};
        this.idf = {};
        this.tfidfMatrix = [];
        this.ready = false;
        this.stopwords = new Set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'about', 'against', 'between', 'through', 'during', 'before',
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'can', 'will', 'just', 'don', 'should', 'now', 'also', 'get', 'got',
            'may', 'would', 'could', 'shall', 'might', 'must', 'need',
        ]);
    }

    async initialize() {
        try {
            const response = await fetch('data.json');
            const data = await response.json();
            this.trainingData = data.training_data;
            this.diseaseInfo = data.disease_info;
            this._buildVocabulary();
            this._computeIDF();
            this._buildTFIDFMatrix();
            this.ready = true;
            console.log(`ML Engine ready: ${this.trainingData.length} samples, ${Object.keys(this.vocabulary).length} features`);
        } catch (err) {
            console.error('Failed to initialize ML engine:', err);
        }
    }

    tokenize(text) {
        return text.toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 1 && !this.stopwords.has(word));
    }

    _buildVocabulary() {
        const docFreq = {};
        this.trainingData.forEach(item => {
            const tokens = new Set(this.tokenize(item.symptoms));
            tokens.forEach(token => {
                docFreq[token] = (docFreq[token] || 0) + 1;
            });
        });
        // Keep terms appearing in at least 2 docs
        let idx = 0;
        for (const [term, freq] of Object.entries(docFreq)) {
            if (freq >= 2) {
                this.vocabulary[term] = idx++;
            }
        }
    }

    _computeIDF() {
        const N = this.trainingData.length;
        const docFreq = {};
        this.trainingData.forEach(item => {
            const tokens = new Set(this.tokenize(item.symptoms));
            tokens.forEach(token => {
                if (token in this.vocabulary) {
                    docFreq[token] = (docFreq[token] || 0) + 1;
                }
            });
        });
        for (const [term, df] of Object.entries(docFreq)) {
            this.idf[term] = Math.log((N + 1) / (df + 1)) + 1;
        }
    }

    _computeTFIDF(text, age, sex, medHistory) {
        // Enrich text with EHR context (same approach as Python model)
        let enriched = text;
        if (age) {
            if (age < 18) enriched += ' pediatric child';
            else if (age < 40) enriched += ' young adult';
            else if (age < 65) enriched += ' middle aged';
            else enriched += ' elderly senior';
        }
        if (sex) enriched += ` ${sex}`;
        if (medHistory && medHistory.length > 0) {
            enriched += ' ' + medHistory.join(' ');
        }

        const tokens = this.tokenize(enriched);
        const tf = {};
        tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
        const maxTf = Math.max(...Object.values(tf), 1);

        const vector = new Float64Array(Object.keys(this.vocabulary).length);
        for (const [term, count] of Object.entries(tf)) {
            if (term in this.vocabulary) {
                const normalizedTf = 0.5 + 0.5 * (count / maxTf);
                vector[this.vocabulary[term]] = normalizedTf * (this.idf[term] || 1);
            }
        }
        return vector;
    }

    _buildTFIDFMatrix() {
        this.tfidfMatrix = this.trainingData.map(item =>
            this._computeTFIDF(item.symptoms)
        );
    }

    _cosineSimilarity(a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom === 0 ? 0 : dot / denom;
    }

    predict(symptoms, topK = 5, age = null, sex = null, medHistory = null) {
        if (!this.ready) return [];

        const queryVec = this._computeTFIDF(symptoms, age, sex, medHistory);

        // Compute similarity to all training samples
        const scores = this.tfidfMatrix.map((vec, i) => ({
            similarity: this._cosineSimilarity(queryVec, vec),
            disease: this.trainingData[i].disease,
        }));

        // Aggregate scores per disease (average of top-K similar samples)
        const diseaseScores = {};
        scores.forEach(s => {
            if (!diseaseScores[s.disease]) diseaseScores[s.disease] = [];
            diseaseScores[s.disease].push(s.similarity);
        });

        const ranked = Object.entries(diseaseScores).map(([disease, sims]) => {
            sims.sort((a, b) => b - a);
            const topSims = sims.slice(0, 3);
            const avg = topSims.reduce((a, b) => a + b, 0) / topSims.length;
            return { disease, score: avg };
        });

        ranked.sort((a, b) => b.score - a.score);

        // Normalize to probabilities using softmax
        const topRanked = ranked.slice(0, topK);
        const maxScore = topRanked[0]?.score || 0;
        const expScores = topRanked.map(r => Math.exp((r.score - maxScore) * 5));
        const sumExp = expScores.reduce((a, b) => a + b, 0);

        return topRanked.map((r, i) => {
            const info = this.diseaseInfo[r.disease] || {};
            return {
                disease: r.disease,
                confidence: expScores[i] / sumExp,
                category: info.category || 'Unknown',
                description: info.description || '',
                severity: info.severity || 'Unknown',
                seek_care: info.seek_care || '',
            };
        });
    }
}

// Global instance
const mlEngine = new SymptomMLEngine();
