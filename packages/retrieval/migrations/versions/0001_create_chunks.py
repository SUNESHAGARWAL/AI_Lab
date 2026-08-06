"""create chunks table

Revision ID: 0001
Revises:
Create Date: 2026-08-06

"""

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGSERIAL PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            document_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks")
