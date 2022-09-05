async def update_flags(self, messages: Sequence[MessageT],
                           flag_set: FrozenSet[Flag], mode: FlagOp) -> None:
        """Update the permanent flags of each messages.

        Args:
            messages: The message objects.
            flag_set: The set of flags for the update operation.
            flag_op: The mode to change the flags.

        """
        ...