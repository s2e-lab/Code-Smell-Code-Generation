def _generate_struct_cstor(self, struct):
        """Emits struct standard constructor."""
        with self.block_func(
                func=self._cstor_name_from_fields(struct.all_fields),
                args=fmt_func_args_from_fields(struct.all_fields),
                return_type='instancetype'):
            for field in struct.all_fields:
                self._generate_validator(field)

            self.emit()

            super_fields = [
                f for f in struct.all_fields if f not in struct.fields
            ]

            if super_fields:
                super_args = fmt_func_args([(fmt_var(f.name), fmt_var(f.name))
                                            for f in super_fields])
                self.emit('self = [super {}:{}];'.format(
                    self._cstor_name_from_fields(super_fields), super_args))
            else:
                if struct.parent_type:
                    self.emit('self = [super initDefault];')
                else:
                    self.emit('self = [super init];')
            with self.block_init():
                for field in struct.fields:
                    field_name = fmt_var(field.name)

                    if field.has_default:
                        self.emit('_{} = {} ?: {};'.format(
                            field_name, field_name, fmt_default_value(field)))
                    else:
                        self.emit('_{} = {};'.format(field_name, field_name))
        self.emit()