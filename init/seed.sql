CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category TEXT CHECK (category IN ('MQL', 'SQL', 'Information Seeker', 'Junk', 'Agent/Bot')),
    lead_email TEXT,
    lead_phone TEXT,
    qa_history JSONB,
    qualifying_qa JSONB
);