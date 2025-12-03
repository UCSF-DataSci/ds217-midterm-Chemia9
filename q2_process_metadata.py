#!/usr/bin/env python3
# Assignment 5, Question 2: Python Data Processing
# Process configuration files for data generation.


def parse_config(filepath: str) -> dict:
    """
    Parse config file (key=value format) into dictionary.

    Args:
        filepath: Path to q2_config.txt

    Returns:
        dict: Configuration as key-value pairs

    Example:
        >>> config = parse_config('q2_config.txt')
        >>> config['sample_data_rows']
        '100'
    """
    # TODO: Read file, split on '=', create dict
    config = {}
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            key, value = line.split('=')
            config[key.strip()] = value.strip()
    return config


def validate_config(config: dict) -> dict:
    """
    Validate configuration values using if/elif/else logic.

    Rules:
    - sample_data_rows must be an int and > 0
    - sample_data_min must be an int and >= 1
    - sample_data_max must be an int and > sample_data_min

    Args:
        config: Configuration dictionary

    Returns:
        dict: Validation results {key: True/False}

    Example:
        >>> config = {'sample_data_rows': '100', 'sample_data_min': '18', 'sample_data_max': '75'}
        >>> results = validate_config(config)
        >>> results['sample_data_rows']
        True
    """
    # TODO: Implement with if/elif/else
    results = {}
    if int(config.get('sample_data_rows')) > 0:
        results['sample_data_rows'] = True
    else:
        results['sample_data_rows'] = False
    if int(config.get('sample_data_min')) >= 1:
        results['sample_data_min'] = True
    else:
        results['sample_data_min'] = False
    if int(config.get('sample_data_max')) > int(config.get('sample_data_min')):
        results['sample_data_max'] = True
    else:
        results['sample_data_max'] = False
    return results


def generate_sample_data(filename: str, config: dict) -> None:
    """
    Generate a file with random numbers for testing, one number per row with no header.
    Uses config parameters for number of rows and range.

    Args:
        filename: Output filename (e.g., 'sample_data.csv')
        config: Configuration dictionary with sample_data_rows, sample_data_min, sample_data_max

    Returns:
        None: Creates file on disk

    Example:
        >>> config = {'sample_data_rows': '100', 'sample_data_min': '18', 'sample_data_max': '75'}
        >>> generate_sample_data('sample_data.csv', config)
        # Creates file with 100 random numbers between 18-75, one per row
        >>> import random
        >>> random.randint(18, 75)  # Returns random integer between 18-75
    """
    # TODO: Parse config values (convert strings to int)
    sample_data_rows = int(config.get('sample_data_rows'))
    sample_data_min = int(config.get('sample_data_min'))
    sample_data_max = int(config.get('sample_data_max'))
    # TODO: Generate random numbers and save to file
    import random
    with open(filename, 'w') as file:
        for n in range(sample_data_rows):
            data = random.randint(sample_data_min, sample_data_max)
            file.write(f"{data}\n")

    # TODO: Use random module with config-specified range


def calculate_statistics(data: list) -> dict:
    """
    Calculate basic statistics.

    Args:
        data: List of numbers

    Returns:
        dict: {mean, median, sum, count}

    Example:
        >>> stats = calculate_statistics([10, 20, 30, 40, 50])
        >>> stats['mean']
        30.0
    """
    # TODO: Calculate stats
    dict = {}
    dict['mean'] = sum(data) / len(data)
    if len(data) % 2 == 0:
        mid1 = data[len(data)//2 - 1]
        mid2 = data[len(data)//2]
        dict['median'] = (mid1 + mid2) / 2
    else:
        dict['median'] = data[len(data)//2]
    dict['sum'] = sum(data)
    dict['count'] = len(data)
    return dict


if __name__ == '__main__':
    # TODO: Test your functions with sample data
    # Example:
    # config = parse_config('q2_config.txt')
    # validation = validate_config(config)
    # generate_sample_data('data/sample_data.csv', config)

    config = parse_config('q2_config.txt')
    validation = validate_config(config)
    print("Validation Results:", validation)
    if all(validation.values()):
        generate_sample_data('data/sample_data.csv', config)
    else:
        print("Invalid configuration. Please check q2_config.txt")

    # 
    # TODO: Read the generated file and calculate statistics
    with open('data/sample_data.csv', 'r') as file:
        data = file.readlines()
        stats = calculate_statistics([int(x) for x in data])
    print(stats)
    # TODO: Save statistics to output/statistics.txt
    with open('output/statistics.txt', 'w') as file:
        print(f"Mean: {stats['mean']:.1f}", file=file)
        print(f"Median: {stats['median']:.1f}", file=file)
        print(f"Sum: {stats['sum']:.1f}", file=file)
        print(f"Count: {stats['count']:.1f}", file=file)