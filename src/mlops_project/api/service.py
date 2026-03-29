from src.mlops_project.api.schema import CustomerInput

def predict(data: CustomerInput) -> float:

    # rule fake => return model.predic_prob(..)
    if data.monthly_charges > 50:
        return 0.7
    else:
        return 0.3