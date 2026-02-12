from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from voice_recognition.core.models import RecognitionScope, Speaker
from voice_recognition.storage.factory import build_repository
from voice_recognition.storage.session_repository import SessionSpeakerRepository
from voice_recognition.storage.sqlite_repository import SQLiteSpeakerRepository


class DualScopeRepositoryTests(unittest.TestCase):
    def test_global_repository_persists_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "speakers.db"
            repo1 = SQLiteSpeakerRepository(db_path)
            created = repo1.create_speaker(
                Speaker(
                    id=None,
                    name="新人1",
                    centroid=(1.0, 0.0, 0.0),
                    prototypes=((1.0, 0.0, 0.0), (0.98, 0.2, 0.0)),
                    sample_count=3,
                )
            )
            repo1.close()

            repo2 = SQLiteSpeakerRepository(db_path)
            speakers = repo2.list_speakers()
            repo2.close()

            self.assertEqual(1, len(speakers))
            self.assertEqual(created.name, speakers[0].name)
            self.assertGreaterEqual(len(speakers[0].all_prototypes()), 2)

    def test_session_repository_is_ephemeral(self) -> None:
        repo1 = SessionSpeakerRepository()
        repo1.create_speaker(
            Speaker(id=None, name="新人1", centroid=(1.0, 0.0, 0.0), sample_count=2)
        )
        self.assertEqual(1, len(repo1.list_speakers()))
        repo1.close()

        repo2 = SessionSpeakerRepository()
        self.assertEqual([], repo2.list_speakers())
        repo2.close()

    def test_factory_returns_expected_implementations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            global_repo = build_repository(RecognitionScope.GLOBAL, db_path=db_path)
            session_repo = build_repository(RecognitionScope.SESSION, db_path=db_path)
            try:
                self.assertIsInstance(global_repo, SQLiteSpeakerRepository)
                self.assertIsInstance(session_repo, SessionSpeakerRepository)
            finally:
                global_repo.close()
                session_repo.close()


if __name__ == "__main__":
    unittest.main()
