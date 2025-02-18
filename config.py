import yaml
import os

package_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(package_dir,"./config.yaml")
with open(file_path, 'r') as yaml_file:
    config = yaml.load(yaml_file, yaml.SafeLoader)
