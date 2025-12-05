from typing import Union

class BankAccount:
    def __init__(self, initial_balance: float = 0.0):
        """
        Initialize a bank account with an optional initial balance.
        
        Args:
            initial_balance (float, optional): Starting balance for the account. Defaults to 0.0.
        """
        self._balance = max(0.0, initial_balance)
        self._transaction_count = 0
    
    def withdraw(self, amount: float) -> float:
        """
        Withdraw a specified amount from the bank account.
        
        Args:
            amount (float): The amount to withdraw.
        
        Returns:
            float: The new account balance after withdrawal.
        
        Raises:
            ValueError: If the withdrawal amount is invalid.
        """
        # Validate withdrawal amount is positive
        if amount <= 0:
            raise ValueError("Withdrawal amount must be a positive number")
        
        # Check if withdrawal amount exceeds current balance
        if amount > self._balance:
            raise ValueError("Insufficient funds: withdrawal amount exceeds current balance")
        
        # Subtract amount from balance
        self._balance -= amount
        
        # Increment transaction count
        self._transaction_count += 1
        
        # Return new balance
        return self._balance
    
    @property
    def balance(self) -> float:
        """
        Get the current account balance.
        
        Returns:
            float: Current account balance.
        """
        return self._balance
    
    @property
    def transaction_count(self) -> int:
        """
        Get the total number of transactions.
        
        Returns:
            int: Number of transactions performed.
        """
        return self._transaction_count
