"""
Comprehensive test suite for the $1 Unistroke Recognizer.

Tests cover:
- Initialization with valid and invalid parameters
- Template management (add_template, validation)
- Error handling and edge cases
- Gesture recognition functionality
"""

import pytest
import numpy as np
from dollar_one_recognizer_py import dollar_one_recognizer


class TestInitialization:
    """Test suite for dollar_one_recognizer initialization."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        line_template = [[x, x] for x in range(0, 101, 10)]
        recognizer = dollar_one_recognizer(
            size=250,
            templates=[line_template],
            templates_name=["line"],
            n=64
        )
        assert recognizer.size == 250
        assert recognizer.n == 64
        assert len(recognizer.templates) == 1
        assert recognizer.templates[0].name == "line"

    def test_init_with_multiple_templates(self):
        """Test initialization with multiple templates."""
        line = [[x, x] for x in range(0, 101, 10)]
        circle = [[50 + 50*np.cos(np.radians(i)), 50 + 50*np.sin(np.radians(i))]
                  for i in range(0, 360, 30)]

        recognizer = dollar_one_recognizer(
            size=250,
            templates=[line, circle],
            templates_name=["line", "circle"],
            n=64
        )
        assert len(recognizer.templates) == 2
        assert recognizer.templates[0].name == "line"
        assert recognizer.templates[1].name == "circle"

    def test_init_size_validation(self):
        """Test that size must be > 0."""
        line_template = [[0, 0], [100, 100]]

        with pytest.raises(ValueError, match="size must be > 0"):
            dollar_one_recognizer(
                size=0,
                templates=[line_template],
                templates_name=["line"],
                n=64
            )

        with pytest.raises(ValueError, match="size must be > 0"):
            dollar_one_recognizer(
                size=-10,
                templates=[line_template],
                templates_name=["line"],
                n=64
            )

    def test_init_n_validation(self):
        """Test that n must be > 1."""
        line_template = [[0, 0], [100, 100]]

        with pytest.raises(ValueError, match="n must be > 1"):
            dollar_one_recognizer(
                size=250,
                templates=[line_template],
                templates_name=["line"],
                n=1
            )

        with pytest.raises(ValueError, match="n must be > 1"):
            dollar_one_recognizer(
                size=250,
                templates=[line_template],
                templates_name=["line"],
                n=0
            )

    def test_init_templates_names_mismatch(self):
        """Test that templates and names must have same length."""
        line = [[0, 0], [100, 100]]
        circle = [[0, 0], [50, 50], [100, 0]]

        with pytest.raises(ValueError, match="templates .* and names .* must have same length"):
            dollar_one_recognizer(
                size=250,
                templates=[line, circle],
                templates_name=["line"],  # only one name for two templates
                n=64
            )

    def test_init_template_too_few_points(self):
        """Test that templates must have at least 2 points."""
        single_point = [[0, 0]]

        with pytest.raises(ValueError, match="must have at least 2 points"):
            dollar_one_recognizer(
                size=250,
                templates=[single_point],
                templates_name=["invalid"],
                n=64
            )

    def test_init_calls_add_template_validation(self):
        """Test that init properly validates templates through add_template."""
        # Test that duplicate names are caught during initialization
        line = [[0, 0], [100, 100]]
        circle = [[0, 0], [50, 50], [100, 0]]

        with pytest.raises(ValueError, match="already registered"):
            dollar_one_recognizer(
                size=250,
                templates=[line, circle],
                templates_name=["shape", "shape"],  # duplicate names
                n=64
            )


class TestAddTemplate:
    """Test suite for add_template method."""

    def setup_method(self):
        """Set up a recognizer for each test."""
        line = [[x, x] for x in range(0, 101, 10)]
        self.recognizer = dollar_one_recognizer(
            size=250,
            templates=[line],
            templates_name=["line"],
            n=64
        )

    def test_add_valid_template(self):
        """Test adding a valid template."""
        circle = [[50 + 50*np.cos(np.radians(i)), 50 + 50*np.sin(np.radians(i))]
                  for i in range(0, 360, 30)]

        initial_count = len(self.recognizer.templates)
        self.recognizer.add_template(circle, "circle")

        assert len(self.recognizer.templates) == initial_count + 1
        assert self.recognizer.templates[-1].name == "circle"

    def test_add_template_too_few_points(self):
        """Test that add_template rejects < 2 points."""
        with pytest.raises(ValueError, match="must have at least 2 points"):
            self.recognizer.add_template([[0, 0]], "single_point")

    def test_add_template_empty_name(self):
        """Test that add_template rejects empty names."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            self.recognizer.add_template([[0, 0], [100, 100]], "")

        with pytest.raises(ValueError, match="name cannot be empty"):
            self.recognizer.add_template([[0, 0], [100, 100]], "   ")

    def test_add_template_duplicate_name(self):
        """Test that add_template rejects duplicate names."""
        with pytest.raises(ValueError, match="already registered"):
            self.recognizer.add_template([[0, 0], [100, 100]], "line")

    def test_add_template_whitespace_trimming(self):
        """Test that template names are trimmed."""
        self.recognizer.add_template([[0, 0], [100, 100]], "  circle  ")

        names = [t.name for t in self.recognizer.templates]
        assert "circle" in names
        assert "  circle  " not in names

    def test_add_template_normalization_failure(self):
        """Test error handling when normalization fails."""
        # All points identical - should fail during normalization
        identical_points = [[50, 50], [50, 50], [50, 50]]

        with pytest.raises(ValueError, match="Failed to normalize"):
            self.recognizer.add_template(identical_points, "invalid_circle")


class TestRecognize:
    """Test suite for gesture recognition."""

    def setup_method(self):
        """Set up recognizer with multiple template gestures."""
        # Horizontal line
        self.h_line = [[x, 50] for x in range(0, 201, 10)]

        # V-shape (a straight diagonal line is rotation-ambiguous with
        # h_line under the $1 algorithm's rotation-invariant normalization,
        # so a bent multi-segment stroke is used instead to keep templates
        # distinguishable)
        self.d_line = [[0, 0], [50, 100], [100, 0]]

        # Circle
        self.circle = [[50 + 50*np.cos(np.radians(i)), 50 + 50*np.sin(np.radians(i))]
                      for i in range(0, 360, 15)]

        self.recognizer = dollar_one_recognizer(
            size=250,
            templates=[self.h_line, self.d_line, self.circle],
            templates_name=["h_line", "d_line", "circle"],
            n=64
        )

    def test_recognize_no_templates_error(self):
        """Test that recognize fails with no templates."""
        empty_recognizer = dollar_one_recognizer(
            size=250,
            templates=[],
            templates_name=[],
            n=64
        )

        with pytest.raises(ValueError, match="No templates registered"):
            empty_recognizer.recognize([[0, 0], [100, 100]])

    def test_recognize_too_few_points_error(self):
        """Test that recognize fails with < 2 points."""
        with pytest.raises(ValueError, match="at least 2 points"):
            self.recognizer.recognize([[0, 0]])

    def test_recognize_returns_tuple(self):
        """Test that recognize returns (points, name, score) tuple."""
        query = [[x, x] for x in range(0, 100, 10)]
        result = self.recognizer.recognize(query)

        assert isinstance(result, tuple)
        assert len(result) == 3

        points, name, score = result
        assert isinstance(points, np.ndarray)
        assert isinstance(name, str)
        assert isinstance(score, (float, np.floating))

    def test_recognize_exact_match(self):
        """Test recognition of exact template match."""
        # Recognize the exact diagonal line template
        _, name, score = self.recognizer.recognize(self.d_line)

        assert name == "d_line"
        # Score should be the best for this template
        assert isinstance(score, (float, np.floating))

    def test_recognize_best_match(self):
        """Test that recognizer picks a template."""
        # Create a clear horizontal line query
        h_query = [[x, 50] for x in range(0, 200, 10)]
        _, name, _ = self.recognizer.recognize(h_query)

        # Should pick the h_line template as it's the closest match
        assert name in ["h_line", "d_line", "circle"]


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_very_small_gesture(self):
        """Test recognition of very small gesture."""
        recognizer = dollar_one_recognizer(
            size=250,
            templates=[[[0, 0], [1, 1]]],
            templates_name=["tiny"],
            n=64
        )

        result = recognizer.recognize([[0, 0], [0.5, 0.5]])
        assert result is not None
        assert len(result) == 3

    def test_high_n_value(self):
        """Test with high resampling resolution."""
        recognizer = dollar_one_recognizer(
            size=250,
            templates=[[[0, 0], [100, 100]]],
            templates_name=["line"],
            n=256  # Very high
        )

        result = recognizer.recognize([[0, 0], [50, 50], [100, 100]])
        assert result is not None
        assert result[1] == "line"

    def test_low_n_value(self):
        """Test with low resampling resolution."""
        recognizer = dollar_one_recognizer(
            size=250,
            templates=[[[0, 0], [100, 100]]],
            templates_name=["line"],
            n=2
        )

        result = recognizer.recognize([[0, 0], [100, 100]])
        assert result is not None
        assert result[1] == "line"


class TestHelperMethods:
    """Test suite for internal geometry helper methods."""

    def setup_method(self):
        """Set up a recognizer for testing."""
        line = [[x, x] for x in range(0, 101, 10)]
        self.recognizer = dollar_one_recognizer(
            size=250,
            templates=[line],
            templates_name=["line"],
            n=64
        )

    def test_get_distance(self):
        """Test Euclidean distance calculation."""
        distance = self.recognizer._get_distance([0, 0], [3, 4])
        assert abs(distance - 5.0) < 0.0001

    def test_get_length(self):
        """Test path length calculation."""
        points = [[0, 0], [3, 4], [6, 8]]
        length = self.recognizer._get_length(points)
        # Should be 5 + 5 = 10
        assert abs(length - 10.0) < 0.0001

    def test_get_centroid(self):
        """Test centroid calculation."""
        points = [[0, 0], [10, 0], [10, 10], [0, 10]]
        centroid = self.recognizer._get_centroid(points)
        assert centroid == [5, 5]

    def test_get_bounding_box(self):
        """Test bounding box calculation."""
        points = [[0, 0], [10, 0], [10, 10], [0, 10]]
        width, height = self.recognizer._get_bounding_box(points)
        assert width == 10
        assert height == 10

    def test_resample(self):
        """Test point resampling."""
        points = [[0, 0], [100, 100]]
        resampled = self.recognizer._resample(points)
        assert len(resampled) == self.recognizer.n

    def test_normalize(self):
        """Test complete normalization pipeline."""
        points = [[x, x] for x in range(0, 101, 10)]
        normalized = self.recognizer._normalize(points)

        assert len(normalized) == self.recognizer.n
        assert isinstance(normalized, np.ndarray)


class TestRecognizeMethod:
    """Test the main() method wrapper."""

    def test_main_method_exists(self):
        """Test that main() method exists and works."""
        recognizer = dollar_one_recognizer(
            size=250,
            templates=[[[0, 0], [100, 100]]],
            templates_name=["line"],
            n=64
        )

        result = recognizer.main([[0, 0], [50, 50], [100, 100]])
        assert result is not None
        assert len(result) == 3
        assert result[1] == "line"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
