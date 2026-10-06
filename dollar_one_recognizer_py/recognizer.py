
# The $1 Unistroke Recognizer (Python 3 version)
#
#     Jacob O. Wobbrock, Ph.D.
#     The Information School
#     University of Washington
#     wobbrock@uw.edu
#
#     Andrew D. Wilson, Ph.D.
#     Microsoft Research
#     awilson@microsoft.com
#
#     Yang Li, Ph.D.
#     Department of Computer Science and Engineering
#     University of Washington
#     yangli@cs.washington.edu
#
# The academic publication for the $1 recognizer, and what should be
# used to cite it, is:
#
#     Wobbrock, J.O., Wilson, A.D. and Li, Y. (2007). Gestures without
#     libraries, toolkits or training: A $1 recognizer for user interface
#     prototypes. Proceedings of the ACM Symposium on User Interface
#     Software and Technology (UIST '07). Newport, Rhode Island (October
#     7-10, 2007). New York: ACM Press, pp. 159-168.
#     https://dl.acm.org/citation.cfm?id=1294238
#
# The Protractor enhancement was separately published by Yang Li and programmed
# here by Jacob O. Wobbrock:
#
#     Li, Y. (2010). Protractor: A fast and accurate gesture
#     recognizer. Proceedings of the ACM Conference on Human
#     Factors in Computing Systems (CHI '10). Atlanta, Georgia
#     (April 10-15, 2010). New York: ACM Press, pp. 2169-2172.
#     https://dl.acm.org/citation.cfm?id=1753654
#
# This software is distributed under the "New BSD License" agreement:
#
# Copyright (C) 2007-2012, Jacob O. Wobbrock, Andrew D. Wilson and Yang Li.
# All rights reserved.
#
# Python 3 port Copyright (C) 2026, yuzujelly2222.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the names of the University of Washington nor Microsoft,
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Jacob O. Wobbrock OR Andrew D. Wilson
# OR Yang Li OR yuzujelly2222 BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Sequence

import numpy as np

Point = Sequence[float]
Points = Sequence[Point]


class GestureTemplate:
    """A single gesture template for matching."""

    points: np.ndarray
    """Normalized template stroke points."""

    name: str
    """Template identifier/display name."""

    def __init__(self, points: np.ndarray, name: str) -> None:
        """Initialize template.
        
        Args:
            points: Pre-normalized point array
            name: Template name
        """
        self.points = points
        self.name = name




class _deg:
    sin = staticmethod(lambda x: np.sin(np.deg2rad(x)))
    cos = staticmethod(lambda x: np.cos(np.deg2rad(x)))
    tan = staticmethod(lambda x: np.tan(np.deg2rad(x)))
    arcsin = staticmethod(lambda x: np.rad2deg(np.arcsin(x)))
    arccos = staticmethod(lambda x: np.rad2deg(np.arccos(x)))
    arctan = staticmethod(lambda x: np.rad2deg(np.arctan(x)))
    arctan2 = staticmethod(lambda y, x: np.rad2deg(np.arctan2(y, x)))


class dolla_one_recognizer:
    """$1 Unistroke Recognizer - Single-stroke gesture recognition.
    
    This is a Python 3 port of the $1 Unistroke Recognizer algorithm
    (Wobbrock, Wilson, and Li, UIST '07).
    
    The recognizer does not require machine learning or training.
    Simply register one template example per gesture shape and it works.
    
    The algorithm:
    1. Resample input stroke to fixed number of points
    2. Rotate to canonical position
    3. Scale and translate to standard size
    4. Match against templates using Golden Section Search
    5. Return best match with confidence score
    
    Reference:
        Wobbrock, J.O., Wilson, A.D. and Li, Y. (2007). Gestures without
        libraries, toolkits or training: A $1 recognizer for user interface
        prototypes. Proceedings of the ACM Symposium on User Interface
        Software and Technology (UIST '07), pp. 159-168.
        https://dl.acm.org/citation.cfm?id=1294238
    
    Attributes:
        size (float): Normalization box size
        n (int): Number of resampled points

    Example:
        >>> from dolla_one_recognizer_py import dolla_one_recognizer
        >>> # Create recognizer with one template
        >>> line = [[x, x] for x in range(0, 101, 5)]
        >>> recognizer = dolla_one_recognizer(
        ...     size=250,
        ...     templates=[line],
        ...     templates_name=["line"],
        ...     n=64
        ... )
        >>> # Recognize a gesture
        >>> query = [[x, x*1.05] for x in range(0, 100, 10)]
        >>> _, name, score = recognizer.recognize(query)
        >>> print(f"{name}: {score:.2%}")
        line: 96.24%
    """
    def __init__(self, size: float, templates: Sequence[Points], templates_name: Sequence[str], n: int) -> None:
        """Initialize the $1 Unistroke Recognizer.

        Args:
            size: Bounding box size for normalization (e.g., 250).
                  Larger values give more precise matching.
            templates: List of template stroke point lists.
                      Each template should have at least 2 points.
            templates_name: Names corresponding to each template.
                           Must match length of templates list.
            n: Number of points after resampling (e.g., 64).
               Higher values are more accurate but slower.

        Raises:
            ValueError: If templates and names lengths don't match,
                       or if size/n are invalid (≤ 0)

        Example:
            >>> line_template = [[x, x] for x in range(0, 101, 5)]
            >>> recognizer = dolla_one_recognizer(
            ...     size=250,
            ...     templates=[line_template],
            ...     templates_name=["line"],
            ...     n=64
            ... )
        """
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        
        if n <= 1:
            raise ValueError(f"n must be > 1, got {n}")
        
        if len(templates) != len(templates_name):
            raise ValueError(
                f"templates ({len(templates)}) and names ({len(templates_name)}) "
                f"must have same length"
            )

        self.size = size
        self.n = n
        self.templates: list[GestureTemplate] = []

        for points, name in zip(templates, templates_name):
            self.add_template(points, name)

    def add_template(self, points: Points, name: str) -> None:
        """Add a new gesture template to the recognizer.

        Args:
            points: Template stroke points [[x, y], ...]
                   Must have at least 2 points.
            name: Unique identifier for this template (e.g., "line", "circle").
                 Must not be empty or already registered.

        Raises:
            ValueError: If points have < 2 elements
            ValueError: If name is empty
            ValueError: If name is already registered
            ValueError: If normalization fails (e.g., all points identical)

        Example:
            >>> recognizer.add_template([[0, 0], [100, 100]], "line")
            >>> recognizer.add_template([[0, 0], [100, 100]], "line")
            ValueError: Template name 'line' is already registered
        """

        if not name or not name.strip():
            raise ValueError("Template name cannot be empty")

        name = name.strip()

        if len(points) < 2:
            raise ValueError(f"Template '{name}' must have at least 2 points, got {len(points)}")


        existing_names = [t.name for t in self.templates]
        if name in existing_names:
            raise ValueError(
                f"Template name '{name}' is already registered. "
                f"Use a different name or remove the existing template first."
            )


        try:
            normalized_points = self._normalize(points)
        except Exception as e:
            raise ValueError(
                f"Failed to normalize template '{name}': {e}. "
                f"Check that points are valid coordinates."
            ) from e
        template = GestureTemplate(normalized_points, name)
        self.templates.append(template)

    def _get_distance(self, p1: Point, p2: Point) -> float:
        """Calculate the Euclidean distance between two points.
    
        Args:
            p1: Point as (x, y) tuple.
            p2: Point as (x, y) tuple.
        
        Returns:
            The Euclidean distance between p1 and p2.
        """
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    def _get_length(self, points: Points) -> float:
        """Calculate the total length of a gesture path.
        
        Args:
            points: List of (x, y) tuples representing the gesture.

        Returns:
            The total path length of the gesture.
        """
        return sum(self._get_distance(points[i-1], points[i]) for i in range(1, len(points)))

    def _get_centroid(self, points: Points) -> list[float]:
        """Calculate the centroid of a gesture path.
                
        Args:
            points: List of (x, y) tuples representing the gesture.
            
        Returns:
            The centroid of the gesture.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [sum(xs)/len(points), sum(ys)/len(points)]

    def _get_bounding_box(self, points: Points) -> tuple[float, float]:
        """Calculate the bounding box of a gesture path.
        
        Args:
            points: List of (x, y) tuples representing the gesture.
            
        Returns:
            Tuple of (width, height) for the bounding box."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_max = max(xs)
        x_min = min(xs)
        y_max = max(ys)
        y_min = min(ys)
        width = abs(x_max - x_min)
        height = abs(y_max - y_min)
        return width, height

    def _all_point_rotate(self, points: Points, theta: float) -> np.ndarray:
        """Rotate all points around their centroid by theta degrees.
    
        Args:
            points: List of (x, y) tuples to rotate.
            theta: Rotation angle in degrees.
        
        Returns:
            Rotated points as numpy array.
        """

        c = self._get_centroid(points)
        new_points = list()
        for point in points:
            qx = (point[0]-c[0])*_deg.cos(theta) - (point[1]-c[1])*_deg.sin(theta) + c[0]
            qy = (point[0]-c[0])*_deg.sin(theta) + (point[1]-c[1])*_deg.cos(theta) + c[1]
            new_points.append([qx,qy])
        return np.array(new_points)

    def _resample(self, points: Points) -> np.ndarray:
        """Resample points to have n evenly distributed points along the path.
    
        Args:
            points: List of (x, y) tuples representing the gesture.
        
        Returns:
            Resampled points as numpy array with self.n points.
        """
        points = [list(p) for p in points]
        ideal = self._get_length(points) / (self.n - 1)
        new_points = [points[0][:]]
        all_distance = 0.0
        i = 1
        while i < len(points):
            distance = self._get_distance(points[i-1], points[i])
            if all_distance + distance >= ideal:
                t = (ideal - all_distance) / distance
                qx = points[i-1][0] + t * (points[i][0] - points[i-1][0])
                qy = points[i-1][1] + t * (points[i][1] - points[i-1][1])
                q = [qx, qy]
                new_points.append(q)
                points.insert(i, q)
                all_distance = 0.0
            else:
                all_distance += distance
            i += 1
        if len(new_points) == self.n - 1:
            new_points.append(points[-1][:])
        return np.array(new_points)

    def _rotate_to_zero(self, points: Points) -> np.ndarray:
        """Rotate points so the first point aligns with the centroid on the x-axis.
    
        Args:
            points: List of (x, y) tuples to rotate.
        
        Returns:
            Rotated points as numpy array.
        """
        c = self._get_centroid(points)
        theta = _deg.arctan2(c[1]-points[0][1],c[0]-points[0][0])
        return self._all_point_rotate(points,-theta)

    def _scale_to_square(self, points: Points) -> np.ndarray:
        """Scale points to fit within a square of size x size.
    
        Args:
            points: List of (x, y) tuples to scale.
        
        Returns:
            Scaled points as numpy array.
        """
        bounding_box = self._get_bounding_box(points)
        width = bounding_box[0] or 1
        height = bounding_box[1] or 1
        new_points = list()
        for point in  points:
            qx = point[0] * (self.size/width)
            qy = point[1] * (self.size/height)
            new_points.append([qx,qy])
        return np.array(new_points)

    def _translate_to_origin(self, points: Points) -> np.ndarray:
        """Translate points so their centroid is at the origin.
    
        Args:
            points: List of (x, y) tuples to translate.
        
        Returns:
            Translated points as numpy array.
        """
        c = self._get_centroid(points)
        new_points = list()
        for point in points:
            qx = point[0] - c[0]
            qy = point[1] - c[1]
            new_points.append([qx,qy])
        return np.array(new_points)

    def _normalize(self, points: Points) -> np.ndarray:
        """Normalize gesture points through resampling, rotation, scaling, and translation.
    
        Args:
            points: List of (x, y) tuples representing the raw gesture.
        
        Returns:
            Normalized points as numpy array.
        """
        points = self._resample(points)
        points = self._rotate_to_zero(points)
        points = self._scale_to_square(points)
        points = self._translate_to_origin(points)
        return points

    def recognize(self, points: Points) -> tuple[np.ndarray, str, float]:
        """Recognize the gesture and return the best matching template.

        This method normalizes the input stroke and compares it against
        all registered templates using Golden Section Search to find the
        best rotation angle.

        Args:
            points: List of (x, y) tuples representing the gesture stroke.
                   Can have any number of points (will be resampled to n points).
                   Example: [[0, 0], [50, 50], [100, 100]]

        Returns:
            A tuple of (template_points, template_name, confidence_score):
            - template_points (np.ndarray): Normalized points of best matching template
            - template_name (str): Name of the matched template
            - confidence_score (float): Similarity score in range [0.0, 1.0]
                                       (1.0 = perfect match, 0.0 = no match)

        Raises:
            ValueError: If no templates are registered or gesture has < 2 points
            ValueError: If normalization fails (e.g., all points are identical)
            RuntimeError: If matching fails unexpectedly (internal error)

        Example:
            >>> recognizer = dolla_one_recognizer(
            ...     size=250,
            ...     templates=[[[0, 0], [100, 100]]],
            ...     templates_name=["line"],
            ...     n=64
            ... )
            >>> query = [[x, x*1.05] for x in range(0, 100, 10)]
            >>> template, name, score = recognizer.recognize(query)
            >>> if score > 0.85:
            ...     print(f"Recognized: {name} (confidence: {score:.2%})")
            ... else:
            ...     print("No good match found")
        """

        if not self.templates:
            raise ValueError("No templates registered. Use add_template() first.")

        if len(points) < 2:
            raise ValueError(f"Gesture must have at least 2 points, got {len(points)}")

        try:
            points = self._normalize(points)
        except Exception as e:
            raise ValueError(f"Failed to normalize gesture: {e}") from e

        best_point = float('inf')
        best_template = self.templates[0].points
        best_template_name = self.templates[0].name

        for template in self.templates:
            temp_point = self._distance_at_best_angle(points, template.points, -45,45,2)
            if temp_point < best_point:
                best_point = temp_point
                best_template = template.points
                best_template_name = template.name

        score = 1 - best_point / (0.5*((self.size**2+self.size**2)**0.5))
        return best_template, best_template_name, score

    def _distance_at_best_angle(self, points: Points, template: Points, theta_a: float, theta_b: float, theta_d: float) -> float:
        """Find the best rotation angle using golden section search.
    
        Args:
            points: Gesture points to match.
            template: Template points to match against.
            theta_a: Lower bound of rotation range in degrees.
            theta_b: Upper bound of rotation range in degrees.
            theta_d: Angle threshold for termination.
        
        Returns:
            Minimum distance found at the best angle.
        """
        GOLDEN_RATIO = 0.5*(-1+(5)**0.5)
        x_1 = GOLDEN_RATIO*theta_a + (1-GOLDEN_RATIO)*theta_b
        f_1 = self._distance_at_angle(points, template, x_1)
        x_2 = GOLDEN_RATIO*theta_b + (1-GOLDEN_RATIO)*theta_a
        f_2 = self._distance_at_angle(points, template, x_2)
        while abs(theta_b-theta_a) > theta_d:
            if f_1 < f_2:
                theta_b = x_2
                x_2 = x_1
                f_2 = f_1
                x_1 = GOLDEN_RATIO*theta_a +(1-GOLDEN_RATIO)*theta_b
                f_1 = self._distance_at_angle(points, template,x_1)
            else:
                theta_a = x_1
                x_1 = x_2
                f_1 = f_2
                x_2 = GOLDEN_RATIO*theta_b + (1-GOLDEN_RATIO)*theta_a
                f_2 = self._distance_at_angle(points, template, x_2)

        return min(f_1,f_2)

    def _distance_at_angle(self, points: Points, template: Points, theta: float) -> float:
        """Calculate distance between points and template rotated by theta.
    
        Args:
            points: Gesture points.
            template: Template points.
            theta: Rotation angle in degrees.
        
        Returns:
            Distance between rotated points and template.
        """
        new_points = self._all_point_rotate(points,theta)
        return self._path_distance(new_points,template)

    def _path_distance(self, a: Points, b: Points) -> float:
        """Calculate the average distance between corresponding points on two paths.
    
        Args:
            a: First set of points.
            b: Second set of points.
        
        Returns:
            Average distance per point.
        """
        d = 0
        for i in range(len(a)):
            d = d + self._get_distance(a[i],b[i])
        return d / len(a)

    def main(self, points: Points) -> tuple[np.ndarray, str, float]:
        """Recognize a gesture (wrapper for recognize method).
    
        Args:
            points: List of (x, y) tuples representing the gesture.
        
        Returns:
            Tuple of (best_template_points, template_name, confidence_score).
        """
        return self.recognize(points)
