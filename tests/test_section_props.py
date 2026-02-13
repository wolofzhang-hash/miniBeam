import unittest

from minibeam.core.section_props import rect_hollow, rect_solid


class TestSectionProps(unittest.TestCase):
    def test_rect_solid_uses_textbook_cy_cz(self):
        props = rect_solid(100.0, 10.0)
        self.assertAlmostEqual(props.c_y, 50.0)
        self.assertAlmostEqual(props.c_z, 5.0)

    def test_rect_hollow_uses_textbook_cy_cz(self):
        props = rect_hollow(120.0, 60.0, 6.0)
        self.assertAlmostEqual(props.c_y, 60.0)
        self.assertAlmostEqual(props.c_z, 30.0)


if __name__ == "__main__":
    unittest.main()
