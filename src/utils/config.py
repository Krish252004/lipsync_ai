import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Convert paths to absolute paths
    base_dir = Path(__file__).parent.parent.parent
    for section in config.get("paths", {}):
        if isinstance(config["paths"][section], str):
            config["paths"][section] = str(base_dir / config["paths"][section])
            
    return config