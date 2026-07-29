import json
from typing import Dict, Any, Optional

class LSystem:
    """Deterministic L-System parser, rewriting engine, and specification container."""

    def __init__(
        self,
        axiom: str,
        rules: Dict[str, str],
        angle: float = 25.0,
        iterations: int = 3,
        step_size: float = 1.0,
        line_width: float = 2.0
    ):
        self.axiom = axiom
        self.rules = rules
        self.angle = float(angle)
        self.iterations = int(iterations)
        self.step_size = float(step_size)
        self.line_width = float(line_width)

    def expand(self, iterations: Optional[int] = None) -> str:
        """Expand the axiom string using production rules for N iterations."""
        n = iterations if iterations is not None else self.iterations
        current = self.axiom
        for _ in range(n):
            next_seq = []
            for char in current:
                next_seq.append(self.rules.get(char, char))
            current = "".join(next_seq)
        return current

    @staticmethod
    def validate_brackets(string: str) -> bool:
        """Check if branch push/pop brackets '[' and ']' are balanced."""
        stack_depth = 0
        for char in string:
            if char == '[':
                stack_depth += 1
            elif char == ']':
                stack_depth -= 1
                if stack_depth < 0:
                    return False
        return stack_depth == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert L-system specification to a dictionary."""
        return {
            "axiom": self.axiom,
            "rules": self.rules,
            "angle": round(self.angle, 2),
            "iterations": self.iterations,
            "step_size": round(self.step_size, 2),
            "line_width": round(self.line_width, 2)
        }

    def to_json(self) -> str:
        """Serialize specification to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LSystem":
        """Instantiate LSystem from dictionary."""
        return cls(
            axiom=data["axiom"],
            rules=data["rules"],
            angle=data.get("angle", 25.0),
            iterations=data.get("iterations", 3),
            step_size=data.get("step_size", 1.0),
            line_width=data.get("line_width", 2.0)
        )

    @classmethod
    def from_json(cls, json_str: str) -> "LSystem":
        """Instantiate LSystem from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return f"LSystem(axiom='{self.axiom}', rules={self.rules}, angle={self.angle}, iter={self.iterations})"
