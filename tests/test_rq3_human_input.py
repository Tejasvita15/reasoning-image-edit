"""Prevent text-decoding recovery from silently changing human score fields."""
import csv
import io
from pathlib import Path
import tempfile
import unittest

from src.rq3_human.__main__ import HEADERS, parse_csv


class HumanCSVTests(unittest.TestCase):
    def parse(self, field=None):
        row = dict.fromkeys(HEADERS, '')
        row.update(reviewer_id='reviewer_2', presentation_order='1', blind_id='review_test',
                   knowledge_required='FALSE', status='ok', consistency_score='4',
                   instruction_score='3', quality_score='5', rationale='Source unchanged; edit partly correct.')
        if field:
            row[field] = 'INVALID_BYTE'
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerow(row)
        raw = stream.getvalue().encode().replace(b'INVALID_BYTE', b'1\xe2Ai15')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ratings.csv'
            path.write_bytes(raw)
            result = parse_csv(path)
            self.assertEqual(path.read_bytes(), raw)
        return result

    def test_optional_note_keeps_invalid_byte_visible_without_editing_raw_file(self):
        rows, warnings = self.parse('ambiguity_note')
        self.assertEqual(rows[0]['ambiguity_note'], r'1\xe2Ai15')
        self.assertEqual(rows[0]['instruction_score'], '3')
        self.assertEqual(warnings[0]['invalid_bytes_hex'], ['e2'])

    def test_invalid_required_fields_are_rejected(self):
        for field in ('blind_id', 'instruction_score', 'rationale'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'outside optional'):
                self.parse(field)

    def test_valid_csv_and_spreadsheet_boolean_case_are_preserved(self):
        rows, warnings = self.parse()
        self.assertEqual(warnings, [])
        self.assertEqual(rows[0]['knowledge_required'], 'FALSE')


if __name__ == '__main__':
    unittest.main()
